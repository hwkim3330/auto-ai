#!/usr/bin/env python3
"""
Remote Control Server for Thinking Actor AGI
==============================================

"모든 것을 원격으로 조작할 수 있는 능력"

HTTP API server for remote computer control via AGI

Endpoints:
- POST /think - Send query to AGI (thinks and acts)
- POST /action - Execute single action
- GET /status - Get agent status
- GET /screenshot - Get current screenshot
- GET /stream - Server-sent events for real-time updates
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import sys
from pathlib import Path
import json
import time
import base64
from io import BytesIO

# Add path
sys.path.append(str(Path(__file__).parent))
from thinking_actor_agi import ThinkingActorAGI

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Global AGI instance
agi = None


def get_agi():
    """Get or create AGI instance"""
    global agi
    if agi is None:
        print("[Server] Initializing Thinking Actor AGI...")
        agi = ThinkingActorAGI(model="qwen2.5:3b")
        print("[Server] AGI ready!")
    return agi


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time()
    })


@app.route('/think', methods=['POST'])
def think():
    """
    Think about query and execute actions

    Request:
    {
        "query": "Open text editor",
        "max_depth": 1,
        "verbose": true
    }

    Response:
    {
        "success": true,
        "result": {...}
    }
    """
    try:
        data = request.json
        query = data.get('query', '')
        max_depth = data.get('max_depth', 1)
        verbose = data.get('verbose', False)

        if not query:
            return jsonify({'error': 'Query required'}), 400

        # Execute thinking + acting
        agent = get_agi()
        result = agent.think_and_act(query, max_depth=max_depth, verbose=verbose)

        return jsonify({
            'success': True,
            'result': result,
            'timestamp': time.time()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/action', methods=['POST'])
def action():
    """
    Execute single action

    Request:
    {
        "type": "click",
        "params": {"x": 100, "y": 200}
    }

    Response:
    {
        "success": true,
        "executed": true
    }
    """
    try:
        data = request.json
        action_type = data.get('type', '')
        params = data.get('params', {})

        if not action_type:
            return jsonify({'error': 'Action type required'}), 400

        agent = get_agi()

        # Create action command
        from thinking_actor_agi import ActionCommand, Action, ActionType

        # Map type string to ActionType
        type_mapping = {
            'click': ActionType.MOUSE_CLICK,
            'move': ActionType.MOUSE_MOVE,
            'type': ActionType.KEYBOARD_TYPE,
            'key': ActionType.KEYBOARD_KEY,
            'wait': ActionType.WAIT,
        }

        if action_type not in type_mapping:
            return jsonify({'error': f'Unknown action type: {action_type}'}), 400

        action = Action(
            type=type_mapping[action_type],
            params=params,
            timestamp=time.time()
        )

        # Execute
        success = agent.agent.act(action)

        return jsonify({
            'success': True,
            'executed': success,
            'action_type': action_type,
            'timestamp': time.time()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/status', methods=['GET'])
def status():
    """
    Get agent status

    Response:
    {
        "agi_ready": true,
        "vision_enabled": true,
        "ncp_neurons": 1096,
        "total_actions": 10,
        "successful_actions": 8
    }
    """
    try:
        agent = get_agi()

        return jsonify({
            'agi_ready': True,
            'vision_enabled': agent.agent.vision.use_real_vision,
            'vision_features': agent.agent.vision.feature_dim,
            'ncp_neurons': agent.agent.ncp.wiring.total_neurons,
            'ncp_synapses': agent.agent.ncp.wiring.total_synapses,
            'total_thoughts': agent.total_thoughts,
            'total_actions': agent.total_actions,
            'successful_actions': agent.successful_actions,
            'timestamp': time.time()
        })

    except Exception as e:
        return jsonify({
            'agi_ready': False,
            'error': str(e)
        }), 500


@app.route('/screenshot', methods=['GET'])
def screenshot():
    """
    Get current screenshot

    Response:
    {
        "success": true,
        "image": "base64_encoded_png",
        "features": [...],
        "timestamp": 123456.789
    }
    """
    try:
        agent = get_agi()

        # Get screenshot and features
        img = agent.agent.vision.capture_screen()
        features = agent.agent.vision.get_current_features()

        # Convert image to base64
        image_b64 = None
        if img is not None:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': image_b64,
            'features': features.tolist(),
            'features_shape': list(features.shape),
            'timestamp': time.time()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/stream', methods=['GET'])
def stream():
    """
    Server-sent events stream for real-time updates

    Usage:
        var eventSource = new EventSource('/stream');
        eventSource.onmessage = function(e) {
            console.log(JSON.parse(e.data));
        };
    """
    def generate():
        """Generate SSE events"""
        # Send initial connection event
        yield f"data: {json.dumps({'event': 'connected', 'timestamp': time.time()})}\n\n"

        # TODO: Stream thinking tokens and actions in real-time
        # For now, just keep connection alive
        while True:
            time.sleep(1)
            yield f"data: {json.dumps({'event': 'heartbeat', 'timestamp': time.time()})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("REMOTE CONTROL SERVER - Starting")
    print("="*70)
    print()
    print("Endpoints:")
    print("  POST /think       - Think and act on query")
    print("  POST /action      - Execute single action")
    print("  GET  /status      - Get agent status")
    print("  GET  /screenshot  - Get current screenshot")
    print("  GET  /stream      - SSE stream for real-time updates")
    print()
    print("="*70)

    # Run server
    app.run(
        host='0.0.0.0',
        port=8888,
        debug=False,
        threaded=True
    )
