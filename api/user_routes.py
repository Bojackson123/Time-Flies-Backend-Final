from flask import Blueprint, request, jsonify, session, make_response, current_app
from flask_login import login_required, current_user, login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from models import db, User, Video
from flask_cors import CORS, cross_origin
import json
import os

users_bp = Blueprint('users', __name__)


@users_bp.route('/is_logged_in')
def is_logged_in():
    if current_user.is_authenticated:
        # User is logged in, return True
        user = current_user
        return jsonify({
            "isLoggedIn": True, 
            "user_id": session['user_id'],
            "fullname": user.fullname,
            "email": user.email,
            "pfp": user.profile_picture
            })
    else:
        # User is not logged in, return False
        return jsonify({"isLoggedIn": False})

# Route to get all users
@users_bp.route('/get_user', methods=['GET'])
@login_required
def get_user():
    user = current_user
    user_details = {
        'id': user.id,
        'fullname': user.fullname,
        'email': user.email,
        'password': user.password,
        'is_admin': user.is_admin
    }
    return jsonify(user_details)


# route for admin to get all users
@users_bp.route('/get_users', methods=['GET'])
@login_required
def get_users():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized access! Only admins can view all users.'}), 403

    users = User.query.all()
    user_list = []
    for user in users:
        user_list.append(
            {
                'id': user.id,
                'fullname': user.first_name,
                'email': user.email,
                'is_admin': user.is_admin
            })
    return jsonify({'users': user_list})



# Route to create a new user
@users_bp.route('/create_user', methods=['POST'])
def create_user():
    data = request.json
    if 'fullname' not in data or 'email' not in data or 'password' not in data or not data['fullname'] or not data['email'] or not data['password']:
        return jsonify({'error': 'Missing required fields'}), 400
    
    user_exists = User.query.filter_by(email=data['email']).first()
    if user_exists:
        # If user exists, return a 401 Unauthorized error
        return jsonify({'error': 'Email already in use'}), 409
    
    # Validate email format
    if '@' not in data['email']:
        return jsonify({'error': 'Invalid email format'}), 422

    # Validate password requirements
    special_chars = "!@#$%^&*_=+-`"
    if len(data['password']) < 6 or not any(char.isdigit() for char in data['password']) or not any(char in special_chars for char in data['password']):
        return jsonify({'error': 'Password must be at least 6 characters long, include at least one number, and one special character (!, @, #, $, %, ^, &, *, _, -, =, +, `, )'}), 423


    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    new_user = User(fullname=data['fullname'], email=data['email'],
                    password=hashed_password, profile_picture='default.png')
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User created successfully'}), 201


# Route to update user details
@users_bp.route('/update_user', methods=['POST'])
@login_required
def update_user():
    user = current_user
    if not user:
        return jsonify({'error': 'User not found'}), 404

    data = request.json
    if data['fullName'] == '' and data['confirmEmail'] == '':
        return jsonify({'error': 'No data provided'}), 400
    
    if 'fullName' in data:
        if data['fullName'] != '':
            user.fullname = data['fullName']
    if 'confirmEmail' in data:
        if data['confirmEmail'] != '':
            user.email = data['confirmEmail']

    db.session.commit()
    return jsonify({'message': 'User updated successfully'})

@users_bp.route('/update_password', methods=['POST'])
@login_required
def update_password():
    user = current_user
    data = request.get_json()
    current_password = data.get('currentPassword')
    new_password = data.get('newPassword')
 

    if not current_password or not new_password:
        return make_response(jsonify({"error": "Both current and new password are required"}), 400)
   
    
    # Verify the current password
    if not check_password_hash(user.password, current_password):
        return make_response(jsonify({"error": "Current password is incorrect"}), 406)
 
    # Update the user's password
    user.password = generate_password_hash(new_password)
 
    # Save the changes to the database
    db.session.commit()

    return jsonify({"message": "Password successfully updated"}), 200

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@users_bp.route('/update_pfp', methods=['POST'])
@login_required
def update_pfp():
    user = current_user
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if not file:
        return jsonify({'error': 'No file selected for upload'}), 400


    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    pfp_path = current_app.config['UPLOAD_FOLDER'] + "/pfp"
    if user.profile_picture != 'default.png':
        os.remove(os.path.join(pfp_path, user.profile_picture))
        
     # Save the file to the uploads folder
    filename = secure_filename(file.filename)
    user.profile_picture = filename
    
    file_path = os.path.join(pfp_path, filename)
    file.save(file_path)


    db.session.commit()

    return jsonify({'message': 'Profile picture updated successfully'}), 201
        

def delete_user_profile_picture(profile_picture_filename):
    if profile_picture_filename == 'default.png':
        return
    # Define the path to the directory where profile pictures are stored
    profile_pictures_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pfp')
    profile_picture_path = os.path.join(profile_pictures_dir, profile_picture_filename)
    # Check if the file exists and delete it
    if os.path.exists(profile_picture_path):
        os.remove(profile_picture_path)
        
def delete_user_videos_and_banners(user):
    # Define the path to the directory where videos are stored
    videos_dir = current_app.config['UPLOAD_FOLDER']
    # Get all videos uploaded by the user
    user_videos = Video.query.filter_by(user_id=user).all()
    for video in user_videos:
        # Delete the video file
        video_path = os.path.join(videos_dir, video.filename)
        if os.path.exists(video_path):
            os.remove(video_path)
        # Delete the banner file
        banner_path = os.path.join(videos_dir, video.banner)
        if os.path.exists(banner_path):
            os.remove(banner_path)
        # Delete the video from the database
        db.session.delete(video)
    # Commit the changes to the database
    db.session.commit()

@users_bp.route('/delete_user', methods=['DELETE'])
@login_required
def delete_user():
    user = current_user
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    delete_user_profile_picture(user.profile_picture)
    delete_user_videos_and_banners(user.id)

    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'User deleted successfully'}), 200


# Route for user login
@users_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    if 'email' not in data or 'password' not in data or not data['email'] or not data['password']:
        return jsonify({'error': 'Missing email or password'}), 400

    user = User.query.filter_by(email=data['email']).first()
    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({'error': 'Invalid email or password'}), 401

    login_user(user)
    session['user_id'] = user.id
    return jsonify({'message': 'Logged in successfully'}), 200


# Route for user logout
@users_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'})
