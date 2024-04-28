from flask import Blueprint, request, jsonify, send_from_directory, current_app, send_file, Response, abort
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from models import db, Video
from joblib import dump, load
import os
import cv2
import re
import shutil
import uuid

import sys
sys.path.append('..')
from analyze_video import analyze_video

def extract_and_save_first_frame(video_path, output_folder):
    """
    Extracts the first frame from a video and saves it as an image.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Path to the folder where the image will be saved.

    Returns:
        str: Filename of the saved image if successful, None otherwise.
    """
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    if success:
        # Construct the output path for the banner image
        banner_filename = os.path.splitext(os.path.basename(video_path))[0] + '_banner.jpg'
        banner_path = os.path.join(output_folder, banner_filename)
        
        # Save the first frame as an image
        cv2.imwrite(banner_path, image)
        cap.release()
        return banner_filename
    else:
        cap.release()
        return None

videos_bp = Blueprint('videos', __name__)

def convert_video_to_frames(video_path, save_dir):
    """
    Converts a video into individual frames and saves them in a directory.

    Args:
        video_path (str): Path to the video file.
        save_dir (str): Path to the directory where the frames will be saved.

    Returns:
        float: Original FPS (frames per second) of the video.
    """
    # Make sure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Determine video FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        # Read a frame
        success, frame = cap.read()

        # If the frame was not successfully read, break the loop
        if not success:
            break

        # Save every frame without trying to simulate 30 FPS
        save_path = os.path.join(save_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(save_path, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f'All frames have been saved to {save_dir}. Total frames: {frame_count}.')
    return original_fps

ALLOWED_VIDEO_MIME_TYPES = set([
    'video/mp4',
    'video/avi',
    'video/mpeg',
    'video/quicktime',  # Add or remove MIME types as necessary
])

@videos_bp.route('/upload', methods=['POST'])
@login_required
def upload_video():
    """
    Endpoint for uploading a video file.

    Returns:
        Response: JSON response containing the status of the upload.
    """
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    title = request.form.get('title', 'Untitled')
    description = request.form.get('description', '')

    if not file:
        return jsonify({'error': 'No file selected for upload'}), 400
    
    # Check MIME type
    if file.content_type not in ALLOWED_VIDEO_MIME_TYPES:
        return jsonify({'error': 'File type not supported'}), 415

    # Generate a unique filename using UUID
    original_filename = secure_filename(file.filename)
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = str(uuid.uuid4()) + file_extension  # Combine UUID and file extension

    
    # Save the file to the uploads folder with the unique filename
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)

    # Extract the first frame and save it as a banner
    banner_filename = extract_and_save_first_frame(file_path, current_app.config['UPLOAD_FOLDER'])

    # Create a new Video object with the unique filename and save it to the database, including the banner
    new_video = Video(title=title, description=description, filename=unique_filename, banner=banner_filename, author=current_user)
    db.session.add(new_video)
    db.session.commit()

    # Return the video ID in the response
    return jsonify({'message': 'Video uploaded successfully', 'video_id': new_video.id}), 201


# Route to download a video
@videos_bp.route('/download/<int:video_id>', methods=['GET'])
@login_required
def download_video(video_id):
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404

    # Check if the video belongs to the current user
    if not current_user.is_admin and video.author != current_user:
        return jsonify({'error': 'You are not authorized to access this video'}), 403

    return send_from_directory(current_app.config['UPLOAD_FOLDER'], video.filename)

@videos_bp.route('/download_results/<int:video_id>/<int:attention>', methods=['GET'])
@login_required
def download_results(video_id, attention):
    video = Video.query.get(video_id)
    if not video or not video.results:
        # Video not found or no results file available
        return jsonify({'error': 'Results file not found'}), 401
    
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'],"ai_results", str(video_id), f"{video_id}_{attention}.0.png")
    print
    if not os.path.exists(file_path):
        # File does not exist
        return jsonify({'error': 'Results file not found'}), 406

    return send_file(file_path, as_attachment=True), 200


# Route to get all videos
@videos_bp.route('/get', methods=['GET'])
@login_required
def get_videos():
    videos = current_user.videos  # Fetch videos associated with the current user
    video_list = []
    for video in videos:
        video_list.append(
            {'id': video.id, 'title': video.title, 'banner': video.banner, 'description': video.description, 'upload_date': video.upload_date})
    return jsonify({'videos': video_list})

@videos_bp.route('/get/<int:video_id>', methods=['GET'])
@login_required
def get_video(video_id):
    # Fetch the video if it belongs to the current user
    video = Video.query.filter_by(id=video_id, user_id=current_user.id).first()

    if not video:
        # If the video doesn't exist or doesn't belong to the current user, return a 404 error
        abort(404, description="Video not found or access is denied.")

    # Assuming you're serving the video file itself or a page with the video embedded
    # You might want to return the file's URL, its metadata, or render a template
    video_data = {
        "id": video.id,
        "title": video.title,
        "description": video.description,
        "filename": video.filename,
        "banner": video.banner,
        "results": video.results,
        "graphJSONData": video.graphJSONData,
    }

    return jsonify(video_data)

@videos_bp.route('/video/<int:video_id>', methods=['GET'])
def serve_video(video_id):
    # Look up the video by ID
    video = Video.query.get_or_404(video_id)

    # Construct the full path to the video file
    video_path = os.path.join('uploads/', video.filename)
    
    range_header = request.headers.get('Range', None)
    if not range_header:
        response = send_file(video_path, mimetype='video/mp4')
        # Add cache-control headers
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    size = os.path.getsize(video_path)
    byte1, byte2 = 0, None

    # Extract range values
    m = re.search('(\d+)-(\d*)', range_header)
    g = m.groups()

    if g[0]: byte1 = int(g[0])
    if g[1]: byte2 = int(g[1])

    length = size - byte1
    if byte2 is not None:
        length = byte2 + 1 - byte1

    data = None
    with open(video_path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, 206, mimetype='video/mp4', content_type='video/mp4', direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    # Add cache-control headers
    rv.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    rv.headers['Pragma'] = 'no-cache'
    rv.headers['Expires'] = '0'

    return rv

@videos_bp.route('/video/<int:video_id>/graph', methods=['GET'])
def serve_video_for_graph(video_id):
   # Look up the video by ID
    video = Video.query.get_or_404(video_id)

    # Construct the full path to the video file
    video_path = f"uploads/720p/720_{video.filename}"
    return send_from_directory(directory='uploads/', path=video.filename)   
    

@videos_bp.route('/analyze/<video_id>')
def analyze(video_id):
    video = Video.query.get(video_id)
    if video:
        # Construct the file path
        #video_title_with_underscores = video.title.replace(" ", "_")
        video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], video.filename)
        print(video_path)
        
        save_dir = f"{video_id}_frames"
        fps = convert_video_to_frames(video_path, save_dir)
        print(fps)
        figure, json_datas = analyze_video(f"{video_id}_frames")
        
        for attention_percent, results in figure.items():
            result_head = f"{video_id}_{attention_percent}.png"
            if not os.path.exists(f"uploads/ai_results/{video_id}"):
                os.makedirs(f"uploads/ai_results/{video_id}")
            figure_path = f"uploads/ai_results/{video_id}/{result_head}"
            results.savefig(figure_path)
        
        video.results = f"{video_id}"
        
        for attention_percent, json_data in json_datas.items():
            json_head = f"{video_id}_{attention_percent}.json"
            if not os.path.exists(f"uploads/graphJSONData/{video_id}"):
                os.makedirs(f"uploads/graphJSONData/{video_id}")
            json_path = f"uploads/graphJSONData/{video_id}/{json_head}"
            with open(json_path, 'w') as json_file:
                json_file.write(json_data)
        
        video.graphJSONData = json_head
        
        # Commit the changes to the database
        db.session.commit()
        
        shutil.rmtree(f"{video_id}_frames")

        return jsonify({'message': "Video analyzed and results updated."}), 200
    else:
        return jsonify({"Video not found."}), 404
    
    
    
 # Route to get graphJSONData by video ID
@videos_bp.route('/get_graph_data/<int:video_id>/<attention>', methods=['GET'])
def get_graph_data(video_id, attention):
    video = Video.query.get(video_id)
    if not video or not video.graphJSONData:
        # Video not found or no graph data file available
        return jsonify({'error': 'Graph data file not found'}), 401
    print(True)
    
    att_float = int(attention) / 100
    print(att_float)    
    # file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "graphJSONData")
    # full_path = os.path.join(file_path, video.graphJSONData)
    # if not os.path.exists(full_path):
    #     # File does not exist
    #     return jsonify({'error': 'Graph data file not found'}), 406
    # print(True)
    print(f'{video_id}_{att_float}.json')
    # No need to open and jsonify the file, just send it directly
    return send_from_directory(directory='uploads/graphJSONData/' + str(video_id), path=f'{video_id}_{attention}.0.json')
    
    

# Route to delete a video
@videos_bp.route('/delete/<int:video_id>', methods=['DELETE'])
@login_required
def delete_video(video_id):
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404

    # Check if the video belongs to the current user
    if video.author != current_user:
        return jsonify({'error': 'You are not authorized to delete this video'}), 403

    # Delete the video file from the uploads directory
    try:
        os.remove(os.path.join(current_app.config['UPLOAD_FOLDER'], video.filename))
        os.remove(os.path.join(current_app.config['UPLOAD_FOLDER'], video.banner))
        if video.results:
            shutil.rmtree(os.path.join(current_app.config['UPLOAD_FOLDER'],"ai_results", str(video.id)))
        if video.graphJSONData:
            shutil.rmtree(os.path.join(current_app.config['UPLOAD_FOLDER'],"graphJSONData", str(video.id)))
    except FileNotFoundError:
        pass  # File doesn't exist, may have already been deleted

    db.session.delete(video)
    db.session.commit()
    return jsonify({'message': 'Video deleted successfully'})
