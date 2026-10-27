import secrets
from functools import wraps

from flask import Blueprint, g, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

import db

# ============================================================
# LOGIN, LOGOUT AND THE SIGNED-IN USER
# ============================================================

auth_blueprint = Blueprint("auth", __name__)

INVALID_LOGIN_MESSAGE = "Invalid username or password."

# Checked when the username does not exist, so an unknown username takes
# about as long as a wrong password and the two cannot be told apart
_UNUSED_HASH = generate_password_hash(secrets.token_urlsafe(16))


def error_response(message, status_code):

    response = jsonify({
        "status": "error",
        "message": message
    })

    if status_code == 401:
        response.headers["WWW-Authenticate"] = "Bearer"

    return response, status_code


def public_user(user):

    # The fields a client may see: never the password hash
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "student_id": user["student_id"]
    }


def bearer_token():

    header = request.headers.get("Authorization", "")

    scheme, _, token = header.partition(" ")

    if scheme.lower() != "bearer" or not token.strip():
        return None

    return token.strip()


@auth_blueprint.route("/auth/login", methods=["POST"])
def login():

    data = request.get_json(silent=True)

    username = data.get("username") if isinstance(data, dict) else None
    password = data.get("password") if isinstance(data, dict) else None

    if not isinstance(username, str) or not isinstance(password, str) \
            or not username.strip() or not password:
        return error_response("Username and password are required.", 400)

    user = db.get_user_by_username(username.strip())

    password_ok = check_password_hash(
        user["password_hash"] if user else _UNUSED_HASH,
        password
    )

    if user is None or not password_ok or not user["is_active"]:
        return error_response(INVALID_LOGIN_MESSAGE, 401)

    token = secrets.token_urlsafe(32)

    expires_at = db.create_token(user["id"], token)

    return jsonify({
        "status": "success",
        "token": token,
        "expires_at": expires_at,
        "user": public_user(user)
    })


# ============================================================
# AUTHORIZATION HELPERS
# ============================================================

def require_auth(*roles):

    # Rejects the request with 401 unless it carries a valid, unexpired
    # Bearer token, and with 403 if roles are given and the user has none of
    # them. The signed-in user is available to the route as g.user.
    def decorator(view):

        @wraps(view)
        def wrapper(*args, **kwargs):

            token = bearer_token()

            user = db.get_user_by_token(token) if token else None

            if user is None:
                return error_response(
                    "Authentication required. Please sign in again.",
                    401
                )

            if roles and user["role"] not in roles:
                return error_response(
                    "You do not have permission to perform this action.",
                    403
                )

            g.user = user
            g.token = token

            return view(*args, **kwargs)

        return wrapper

    return decorator


def can_access_student(user, student_id):

    # Faculty may access every student; a student only their own record.
    # Decided here on the server from the token's user, never from the client.
    if user["role"] == "faculty":
        return True

    return user["role"] == "student" and user["student_id"] == student_id


def forbidden():

    return error_response(
        "You do not have permission to access this student's data.",
        403
    )


@auth_blueprint.route("/auth/logout", methods=["POST"])
@require_auth()
def logout():

    db.delete_token(g.token)

    return jsonify({
        "status": "success",
        "message": "Signed out."
    })


@auth_blueprint.route("/auth/me", methods=["GET"])
@require_auth()
def me():

    return jsonify({
        "status": "success",
        "expires_at": g.user["expires_at"],
        "user": public_user(g.user)
    })
