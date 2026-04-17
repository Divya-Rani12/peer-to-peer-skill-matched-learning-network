# app.py
from flask_socketio import SocketIO, emit, join_room
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from bson import ObjectId
from flask_mail import Mail, Message
import secrets


load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY")



socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Mongo config


app.config["MONGO_URI"] = os.environ.get("MONGO_URI")
mongo = PyMongo(app)
users_collection = mongo.db.users

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")  # default
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ------- helper functions -------
def embed_text(text: str):
    """
    SAFE embedding fallback.
    If OpenAI key fails, generate a local numeric vector.
    """

    try:
        response = client.embeddings.create(
            model=OPENAI_MODEL,
            input=text
        )
        return response.data[0].embedding

    except Exception as e:
        print("⚠️ OpenAI unavailable, using local embedding:", e)

        # LOCAL fallback embedding (fixed-length vector)
        np.random.seed(abs(hash(text)) % (10**8))
        return np.random.rand(384).tolist()


def build_profile_text(user_doc):
    """
    Create a single text string summarizing a user profile for embedding.
    Combine skills, learn_skills, experience and bio.
    """
    parts = []
    if user_doc.get("first_name") or user_doc.get("last_name"):
        parts.append(f"{user_doc.get('first_name','')} {user_doc.get('last_name','')}")
    if user_doc.get("skills"):
        parts.append("Skills: " + ", ".join(user_doc.get("skills")))
    if user_doc.get("learn_skills"):
        parts.append("Wants to learn: " + ", ".join(user_doc.get("learn_skills")))
    if user_doc.get("experience"):
        parts.append("Experience: " + str(user_doc.get("experience")))
    if user_doc.get("bio"):
        parts.append("Bio: " + user_doc.get("bio"))
    txt = " | ".join([p for p in parts if p])
    return txt if txt else (user_doc.get("email") or "user")

def ensure_user_embedding(email):
    """
    Ensure that a user has an embedding stored (field: embedding).
    If not, compute from profile text and store it into Mongo.
    Returns embedding as numpy array.
    """
    user = users_collection.find_one({"email": email})
    if not user:
        return None
    if user.get("embedding"):
        return np.array(user["embedding"], dtype=float)
    # Build profile text and compute embedding
    text = build_profile_text(user)
    emb = embed_text(text)
    # store embedding in DB
    users_collection.update_one({"email": email}, {"$set": {"embedding": emb}})
    return np.array(emb, dtype=float)

def compute_matches_for(email, top_n=5):
    """
    Compute top-N peer matches for a user identified by email.
    Returns list of dicts: [{email,name,score,skills,bio}, ...]
    """
    target_emb = ensure_user_embedding(email)
    if target_emb is None:
        return []

    all_users = list(users_collection.find({"email": {"$ne": email}}))
    embeddings = []
    others = []

    for u in all_users:
        if u.get("embedding"):
            embeddings.append(np.array(u["embedding"], dtype=float))
            others.append(u)
        else:
            txt = build_profile_text(u)
            try:
                emb = embed_text(txt)
                users_collection.update_one(
                    {"_id": u["_id"]},
                    {"$set": {"embedding": emb}}
                )
                embeddings.append(np.array(emb, dtype=float))
                others.append(u)
            except Exception as e:
                print("Embedding failed for user:", u.get("email"), e)

    if not embeddings:
        return []

    emb_matrix = np.vstack(embeddings)
    sims = cosine_similarity([target_emb], emb_matrix)[0]

    scored = []

    # 🔹 get learner skills once
    current_user = users_collection.find_one({"email": email})
    learner_skills = current_user.get("learn_skills", []) if current_user else []

    for u, s in zip(others, sims):
        skill_score = skill_match_score(
            learner_skills,
            u.get("skills", [])
        )
        final_score = (0.6 * float(s)) + (0.4 * skill_score)

        # fetch rating for this user
        feedbacks = list(mongo.db.feedback.find({"to": u.get("email")}))
        avg_rating = round(sum(f["rating"] for f in feedbacks) / len(feedbacks), 1) if feedbacks else 0

        scored.append({
            "email": u.get("email"),
            "name": f"{u.get('first_name', '')} {u.get('last_name', '')}".strip(),
            "score": final_score,
            "skills": u.get("skills", []),
            "bio": u.get("bio", ""),
            "profile_pic": u.get("profile_pic", None),
            "avg_rating": avg_rating,
            "rating_count": len(feedbacks)
        })

    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored_sorted[:top_n]


def calculate_progress(email):
    feedbacks = list(mongo.db.feedback.find({"to": email}))

    if not feedbacks:
        return 0

    avg = sum(f["rating"] for f in feedbacks) / len(feedbacks)
    return round(avg, 2)

def skill_match_score(learner_skills, mentor_skills):
    if not learner_skills or not mentor_skills:
        return 0.0

    learner = set([s.lower() for s in learner_skills])
    mentor = set([s.lower() for s in mentor_skills])

    matched = learner.intersection(mentor)
    return len(matched) / max(len(learner), 1)


# ------- routes (signup/login/profile/dashboard unchanged) -------

@app.route("/")
def index():
    user_email = session.get("email")
    user = users_collection.find_one({"email": user_email}) if user_email else None
    return render_template("index.html", user=user)

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        first_name = request.form.get("first_name","").strip()
        last_name = request.form.get("last_name","").strip()
        email = request.form.get("email","").strip().lower()
        phone = request.form.get("phone","").strip()
        password = request.form.get("password","")
        if users_collection.find_one({"email": email}):
            flash("Email already registered", "error")
            return redirect(url_for("login"))
        hashed = generate_password_hash(password)
        users_collection.insert_one({
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": phone,
            "password": hashed,
            "role": "user",
            "skills": [],
            "learn_skills": [],
            "experience": "",
            "bio": "",
            "availability": "",
            "language":""
        })
        flash("Account created. Log in.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # ✅ Use mongo.db.users (consistent)
        user = mongo.db.users.find_one({"email": email})

        if not user or not check_password_hash(user["password"], password):
            flash("Invalid email or password", "error")
            return redirect(url_for("login"))

        session["email"] = email
        flash("Logged in", "success")

        # 🔥 ROLE-BASED REDIRECT
        if user.get("role") == "admin":
            return redirect(url_for("admin"))   # 👉 Admin dashboard
        else:
            return redirect(url_for("dashboard"))  # 👉 Normal user dashboard

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "email" not in session:
        return redirect(url_for("login"))

    # ✅ Get logged-in user
    user = mongo.db.users.find_one({"email": session["email"]})

    if not user:
        session.clear()
        return redirect(url_for("login"))

    # ✅ Progress calculation
    try:
        progress = calculate_progress(session["email"])
    except Exception as e:
        print("❌ Progress error:", e)
        progress = 0

    # ✅ LANGUAGE MATCHING (NEW FEATURE)
    user_lang = user.get("language")

    matched_users = []
    if user_lang:
        matched_users = list(mongo.db.users.find({
            "language": user_lang,
            "email": {"$ne": user["email"]}  # exclude self
        }))

    print("✅ Dashboard loaded for:", session["email"])

    return render_template(
        "dashboard.html",
        user=user,
        progress=progress,
        current_user=session["email"],
        matched_users=matched_users   # ✅ PASS THIS
    )
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "email" not in session:
        return redirect(url_for("login"))

    user_email = session["email"]

    if request.method == "POST":
        skills = [s.strip() for s in request.form.get("skills", "").split(",") if s.strip()]
        learn_skills = [s.strip() for s in request.form.get("learn_skills", "").split(",") if s.strip()]
        experience = request.form.get("experience", "")
        bio = request.form.get("bio", "")
        availability = request.form.get("availability", "")
        language = request.form.get("language", "")

        update_data = {
            "skills": skills,
            "learn_skills": learn_skills,
            "experience": experience,
            "bio": bio,
            "availability": availability,
            "language": language   # ✅ ADD THIS
}

        # Handle profile picture upload
        if "profile_pic" in request.files:
            file = request.files["profile_pic"]
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                update_data["profile_pic"] = filename

        # Save updated data
        users_collection.update_one({"email": user_email}, {"$set": update_data})

        # Recompute embedding
        try:
            user_doc = users_collection.find_one({"email": user_email})
            profile_text = build_profile_text(user_doc)
            emb = embed_text(profile_text)
            users_collection.update_one({"email": user_email}, {"$set": {"embedding": emb}})
        except Exception as e:
            print("Embedding error:", e)

        flash("Profile updated successfully!", "success")
        return redirect(url_for("dashboard"))

    # GET request loads the form with existing values
    user = users_collection.find_one({"email": user_email})
    return render_template("profile.html", user=user)


@app.route("/peers")
def peers():
    if "email" not in session:
        return redirect(url_for("login"))

    try:
        user_email = session["email"]

        print("👉 USER:", user_email)

        matches = compute_matches_for(user_email, top_n=10) or []
        print("👉 MATCHES:", matches)

        session_map = {}

        sessions = mongo.db.sessions.find({
            "$or": [
                {"from": user_email},
                {"to": user_email}
            ]
        })

        for s in sessions:
            print("👉 SESSION DOC:", s)

            from_user = s.get("from")
            to_user = s.get("to")

            if not from_user or not to_user:
                print("⚠️ Skipping bad session")
                continue

            other_user = to_user if from_user == user_email else from_user
            session_map[other_user] = s.get("status", "unknown")

        print("👉 SESSION MAP:", session_map)

        return render_template(
            "peers.html",
            matches=matches,
            session_map=session_map,
            current_user=user_email
        )

    except Exception as e:
        print("❌ ERROR IN /peers:", e)
        return f"Error: {e}"


@app.route("/requests")
def requests_page():
    if "email" not in session:
        return redirect(url_for("login"))

    user = session["email"]

    # 🔥 ONLY RECEIVED REQUESTS
    requests = list(mongo.db.sessions.find({
       "$or": [
            {"from": user},
            {"to": user}
        ]
    }))


    return render_template("requests.html", requests=requests)

     # make sure this is already imported

@app.route("/accept-request/<request_id>", methods=["POST"])
def accept_request(request_id):
    if "email" not in session:
        return redirect(url_for("login"))

    # 🔥 CREATE UNIQUE ROOM ID
    room_id = str(ObjectId())

    # 🔥 UPDATE SESSION WITH ROOM ID
    mongo.db.sessions.update_one(
        {"_id": ObjectId(request_id)},
        {"$set": {
            "status": "accepted",
            "room_id": room_id
        }}
    )

    flash("Session accepted! You can now join the meeting.", "success")
    return redirect(url_for("requests_page"))

    

@app.route("/reject-request/<request_id>", methods=["POST"])
def reject_request(request_id):
    if "email" not in session:
        return redirect(url_for("login"))

    mongo.db.sessions.update_one(
    {"_id": ObjectId(request_id)},
        {"$set": {"status": "rejected"}}
    )

    flash("Session request rejected.", "info")
    return redirect(url_for("requests_page"))



@app.route("/request-session", methods=["POST"])
def request_session():
    if "email" not in session:
        return redirect(url_for("login"))
    from_user = session["email"]
    to_user = request.form["peer_email"]

    mongo.db.sessions.insert_one({
        "from": from_user,
        "to": to_user,
        "status": "pending",
        "room_id": None,
        "scheduled_time": request.form.get("scheduled_time"),
        "created_at": datetime.now()
    })

    flash("Session request sent successfully!", "success")  
    return redirect(url_for("peers"))
  # add this at top if not present

@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    if "email" not in session:
        return redirect(url_for("login"))

    mongo.db.feedback.insert_one({
        "session_id": ObjectId(request.form["session_id"]),
        "from": session["email"],
        "to": request.form["to_email"],
        "rating": int(request.form["rating"]),
        "comment": request.form["comment"],
        "created_at": datetime.now()
    })

    flash("Feedback submitted successfully!", "success")
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("index"))

# small API endpoint to return JSON matches (useful for AJAX)
@app.route("/api/matches")
def api_matches():
    if "email" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    matches = compute_matches_for(session["email"], top_n=20) 
    return jsonify({"matches": matches})

@app.route("/meeting/<room_id>")
def meeting(room_id):
    session_doc = mongo.db.sessions.find_one({"room_id": room_id})

    session_id = str(session_doc["_id"]) if session_doc else ""

    user_email = session.get("email")

    if session_doc:
        if session_doc.get("from") == user_email:
            to_email = session_doc.get("to")
        else:
            to_email = session_doc.get("from")
    else:
        to_email = ""

    return render_template(
        "meeting.html",
        room_id=room_id,
        session_id=session_id,
        to_email=to_email
    )

@app.route("/notes/<session_id>", methods=["GET", "POST"])
def notes(session_id):
    if "email" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        mongo.db.notes.update_one(
            {"session_id": ObjectId(session_id)},
            {"$set": {"content": request.form["content"]}},
            upsert=True
        )

    note = mongo.db.notes.find_one({"session_id": ObjectId(session_id)})
    return render_template(
        "notes.html",
        session_id=session_id,
        content=note["content"] if note else ""
    )


@app.route("/schedule")
def schedule():
    if "email" not in session:
        return redirect(url_for("login"))

    user = session["email"]

    # 🔥 ONLY ACCEPTED SESSIONS
    sessions = list(mongo.db.sessions.find({
        "$or": [
            {"from": user},
            {"to": user}
        ],
        "status": "accepted"
    }))


    return render_template("schedule.html", sessions=sessions, user=user)

@app.route("/set-time", methods=["POST"])
def set_time():
    if "email" not in session:
        return redirect(url_for("login"))
    room_id = request.form.get("room_id")
    scheduled_time = request.form.get("scheduled_time")

    mongo.db.sessions.update_one(
        {"room_id": room_id},
        {"$set": {"scheduled_time": scheduled_time}}
    )

    return redirect("/schedule")

@app.route("/admin")
def admin():
    if "email" not in session:
        return redirect(url_for("login"))

    # ✅ FIXED HERE
    user = mongo.db.users.find_one({"email": session["email"]})

    if not user or user.get("role") != "admin":
        flash("Access Denied: Admins only", "error")
        return redirect(url_for("dashboard"))

    users = list(mongo.db.users.find())
    sessions = list(mongo.db.sessions.find())
    feedbacks = list(mongo.db.feedback.find())

    # 📊 METRICS
    dau = len(users)
    matches_today = len(sessions)

    avg_rms = 75
    completion_rate = 80

    active_sessions = len([s for s in sessions if s.get("status") == "active"])

    # 🚨 ALERTS
    low_rated = len([f for f in feedbacks if int(f.get("rating", 0)) <= 2])
    reported_users = 0
    suspicious = 0

    # 📈 GRAPH DATA
    dates = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    user_counts = [5, 10, 15, 20, 25]
    rms_scores = [60, 65, 70, 75, 80]

    return render_template(
        "admin_dashboard.html",
        users=users,
        sessions=sessions,
        feedbacks=feedbacks,
        dau=dau,
        matches_today=matches_today,
        avg_rms=avg_rms,
        completion_rate=completion_rate,
        active_sessions=active_sessions,
        low_rated=low_rated,
        reported_users=reported_users,
        suspicious=suspicious,
        dates=dates,
        user_counts=user_counts,
        rms_scores=rms_scores
    )

@app.route("/test-ai")
def test_ai():
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL"),
        messages=[
            {"role": "user", "content": "Hello"}
        ]
    )
    
    print("OpenAI working") 

    return response.choices[0].message.content    

@app.route("/delete-user/<id>", methods=["POST"])
def delete_user(id):
    mongo.db.users.delete_one({"_id": ObjectId(id)})
    return redirect(url_for("admin"))


@app.route("/delete-session/<id>", methods=["POST"])
def delete_session(id):
    mongo.db.sessions.delete_one({"_id": ObjectId(id)})
    return redirect(url_for("admin"))  

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("message")

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL"),
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    reply = response.choices[0].message.content

    return render_template("chat.html", reply=reply) 

@app.route("/forgot-password")
def forgot_password():
    return render_template("forgot_password.html")


# JOIN ROOM
@socketio.on("join")
def on_join(data):
    room = data["room"]
    join_room(room)
    print("User joined room:", room)

    emit("user-joined", {"id": request.sid}, room=room, include_self=False)

# # 📝 NOTES
@socketio.on("notes-update")
def on_notes_update(data):
    emit("notes-update", data, room=data["room"], include_self=False)
if __name__ == "__main__":
    try:
        mongo.cx.server_info()
        print("MongoDB connected OK")
    except Exception as e:
        print("MongoDB connection problem:", e)

    port = int(os.environ.get("PORT", 5001))

    socketio.run(app, host="0.0.0.0", port=port)