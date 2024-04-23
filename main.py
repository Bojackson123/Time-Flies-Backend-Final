from flask import Flask, send_from_directory
from flask_login import LoginManager
from user_routes import users_bp
from video_routes import videos_bp
from models import db, User
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)


app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True

app.config.from_object('config.Config')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

pfp_route = app.config['UPLOAD_FOLDER'] + "/pfp"

@app.route('/uploads/pfp/<filename>')
def uploaded_pfp(filename):
    return send_from_directory(pfp_route, filename)

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


app.register_blueprint(users_bp, url_prefix='/users')
app.register_blueprint(videos_bp, url_prefix='/videos')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
