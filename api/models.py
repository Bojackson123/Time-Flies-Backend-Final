from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    profile_picture = db.Column(db.String(255), nullable=True, default='default.png')
    # Define a relationship to videos
    videos = db.relationship('Video', backref='author', lazy=True)


# Video model
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    upload_date = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    filename = db.Column(db.String(255), nullable=False)
    banner = db.Column(db.String(255), nullable=True)
    results = db.Column(db.String(255), nullable=True)
    graphJSONData = db.Column(db.String(255), nullable=True)
    # Foreign key to link the video to its author (user)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
