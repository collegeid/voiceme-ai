
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, Namespace
import main
import requests  # Add this import to send HTTP requests

app = Flask(__name__)

# Define the custom namespace
voice_namespace = Namespace('user', description='AI VoiceMe API for voice processing and user authentication')

# Set the base URL and disable swagger.json
api = Api(
    app,
    version='1.0',
    title='AI VoiceMe API',
    description='An API for voice processing and user authentication',
    doc='/docs',  # Swagger UI will be available at /docs instead of default /swagger
    prefix='/api',  # Set base URL for the API to '/api'
    serve_spec=False  # Disable the /swagger.json route
)

# Add the custom namespace to the API
api.add_namespace(voice_namespace)

# Define parsers for each endpoint
register_parser = api.parser()
register_parser.add_argument('username', type=str, required=True, location='form', help='The username of the user')
register_parser.add_argument('password', type=str, required=True, location='form', help='The password of the user')

login_parser = api.parser()
login_parser.add_argument('username', type=str, required=True, location='form', help='The username of the user')
login_parser.add_argument('password', type=str, required=True, location='form', help='The password of the user')

upload_parser = api.parser()
upload_parser.add_argument('file', type='file', required=True, location='files', help='The audio file to upload')

history_parser = api.parser()
history_parser.add_argument('user_id', type=str, required=True, location='args', help='User ID for fetching history')

delete_history_parser = api.parser()
delete_history_parser.add_argument('user_id', type=str, required=True, location='form', help='User ID for deleting history')
delete_history_parser.add_argument('audio_id', type=str, required=True, location='form', help='Audio ID for deleting history')

# Endpoint for registering users
@voice_namespace.route('/register')
class RegisterUser(Resource):
    @voice_namespace.expect(register_parser)
    def post(self):
        try:
            args = register_parser.parse_args()
            username = args['username']
            password = args['password']

            # Send data to another server on port 8501
            response = requests.post('http://localhost:8501/register', data={'username': username, 'password': password})

            if response.status_code == 200:
                return jsonify({"message": "User registered successfully", "data": response.json()}), 201
            else:
                return jsonify({"error": "Failed to register user", "details": response.text}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400

# Endpoint for user login
@voice_namespace.route('/login')
class LoginUser(Resource):
    @voice_namespace.expect(login_parser)
    def post(self):
        try:
            args = login_parser.parse_args()
            username = args['username']
            password = args['password']

            # Send data to another server on port 8501
            response = requests.post('http://localhost:8501/login', data={'username': username, 'password': password})

            if response.status_code == 200:
                return jsonify({"message": "Login successful", "token": response.json().get('token')}), 200
            else:
                return jsonify({"error": "Login failed", "details": response.text}), 401
        except Exception as e:
            return jsonify({"error": str(e)}), 401

# Endpoint for uploading audio files
@voice_namespace.route('/upload')
class UploadAudio(Resource):
    @voice_namespace.expect(upload_parser)
    def post(self):
        try:
            args = upload_parser.parse_args()
            file = request.files.get('file')

            if not file:
                return jsonify({"error": "No file provided"}), 400

            # Send file to another server on port 8501
            files = {'file': file}
            response = requests.post('http://localhost:8501/upload', files=files)

            if response.status_code == 200:
                return jsonify({"message": "File uploaded successfully", "result": response.json()}), 200
            else:
                return jsonify({"error": "File upload failed", "details": response.text}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400

# Endpoint for retrieving audio history
@voice_namespace.route('/history')
class GetHistory(Resource):
    @voice_namespace.expect(history_parser)
    def get(self):
        try:
            args = history_parser.parse_args()
            user_id = args['user_id']

            # Send request to another server on port 8501 to get history
            response = requests.get(f'http://localhost:8501/history', params={'user_id': user_id})

            if response.status_code == 200:
                return jsonify({"data": response.json()}), 200
            else:
                return jsonify({"error": "Failed to retrieve history", "details": response.text}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400

# Endpoint for deleting audio history
@voice_namespace.route('/history/delete')
class DeleteHistory(Resource):
    @voice_namespace.expect(delete_history_parser)
    def delete(self):
        try:
            args = delete_history_parser.parse_args()
            user_id = args['user_id']
            audio_id = args['audio_id']

            # Send request to another server on port 8501 to delete history
            response = requests.delete(f'http://localhost:8501/history/delete', data={'user_id': user_id, 'audio_id': audio_id})

            if response.status_code == 200:
                return jsonify({"message": "History deleted successfully"}), 200
            else:
                return jsonify({"error": "Failed to delete history", "details": response.text}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print(app.url_map)  # Print available routes
    app.run(port=8502, debug=True)  # Running Flask on port 8502
