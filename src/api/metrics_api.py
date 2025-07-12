#!/usr/bin/env python3
"""
Metrics API
===========
API endpoints for serving agent metrics to the website dashboard.
"""

import json
import time
from typing import Dict, List, Any
from datetime import datetime
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Use the correct import path for the root-level metrics tracker
from agent_metrics_tracker import get_metrics_tracker, AgentMetrics

class MetricsAPI:
    """API for serving agent metrics"""
    
    def __init__(self):
        self.tracker = get_metrics_tracker()
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            # Get base metrics from tracker
            base_metrics = self.tracker.get_system_metrics()
            
            # Add additional system info
            enhanced_metrics = {
                **base_metrics,
                'uptime': self._calculate_uptime(),
                'last_updated': time.time(),
                'version': '1.0.0',
                'api_status': 'operational'
            }
            
            return enhanced_metrics
            
        except Exception as e:
            print(f"Error getting system metrics: {e}")
            return {
                'total_agents': 0,
                'active_agents': 0,
                'total_requests': 0,
                'completed_collaborations': 0,
                'failed_collaborations': 0,
                'success_rate': 0.0,
                'status': 'error',
                'uptime': 'Unknown',
                'last_updated': time.time(),
                'error': str(e)
            }
    
    def get_agent_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics for all agents"""
        try:
            agents = self.tracker.get_all_agent_metrics()
            
            # Convert to dict format for JSON serialization
            agent_list = []
            for agent in agents:
                agent_dict = {
                    'id': agent.agent_id,
                    'name': agent.agent_name,
                    'type': agent.agent_type,
                    'status': agent.status,
                    'total_requests': agent.total_requests,
                    'completed_tasks': agent.completed_tasks,
                    'failed_tasks': agent.failed_tasks,
                    'success_rate': agent.success_rate,
                    'collaboration_count': agent.collaboration_count,
                    'last_activity': self._format_last_activity(agent.last_activity),
                    'avg_task_duration': round(agent.avg_task_duration, 2) if agent.avg_task_duration else 0.0
                }
                agent_list.append(agent_dict)
            
            return agent_list
            
        except Exception as e:
            print(f"Error getting agent metrics: {e}")
            return []
    
    def get_agent_metrics_by_id(self, agent_id: str) -> Dict[str, Any]:
        """Get metrics for a specific agent"""
        try:
            agent = self.tracker.get_agent_metrics(agent_id)
            if not agent:
                return {'error': 'Agent not found'}
            
            return {
                'id': agent.agent_id,
                'name': agent.agent_name,
                'type': agent.agent_type,
                'status': agent.status,
                'total_requests': agent.total_requests,
                'completed_tasks': agent.completed_tasks,
                'failed_tasks': agent.failed_tasks,
                'success_rate': agent.success_rate,
                'collaboration_count': agent.collaboration_count,
                'last_activity': self._format_last_activity(agent.last_activity),
                'avg_task_duration': round(agent.avg_task_duration, 2) if agent.avg_task_duration else 0.0
            }
            
        except Exception as e:
            print(f"Error getting agent metrics for {agent_id}: {e}")
            return {'error': str(e)}
    
    def register_agent_activity(self, agent_id: str, activity_type: str, task_id: str = None, **kwargs):
        """Register an agent activity"""
        try:
            if activity_type == 'task_start':
                self.tracker.track_task_start(agent_id, task_id, kwargs.get('task_info', {}))
            elif activity_type == 'task_complete':
                self.tracker.track_task_completion(
                    agent_id, 
                    task_id, 
                    success=kwargs.get('success', True),
                    result=kwargs.get('result')
                )
            elif activity_type == 'agent_register':
                self.tracker.register_agent(
                    agent_id,
                    kwargs.get('agent_name', agent_id),
                    kwargs.get('agent_type', 'Unknown')
                )
            elif activity_type == 'status_update':
                self.tracker.update_agent_status(agent_id, kwargs.get('status', 'online'))
            
            return {'success': True}
            
        except Exception as e:
            print(f"Error registering activity: {e}")
            return {'error': str(e)}
    
    def _calculate_uptime(self) -> str:
        """Calculate system uptime (mock implementation)"""
        # This would normally track actual system start time
        # For now, return a reasonable uptime
        days = 3
        hours = 14
        minutes = 22
        return f"{days} days, {hours} hours, {minutes} minutes"
    
    def _format_last_activity(self, timestamp: float) -> str:
        """Format last activity timestamp as human-readable string"""
        if not timestamp:
            return "Never"
        
        try:
            current_time = time.time()
            diff = current_time - timestamp
            
            if diff < 60:
                return f"{int(diff)} seconds ago"
            elif diff < 3600:
                return f"{int(diff / 60)} minutes ago"
            elif diff < 86400:
                return f"{int(diff / 3600)} hours ago"
            else:
                return f"{int(diff / 86400)} days ago"
                
        except Exception:
            return "Unknown"

# Global API instance
metrics_api = MetricsAPI()

def get_metrics_api() -> MetricsAPI:
    """Get the global metrics API instance"""
    return metrics_api

# Simple HTTP server for testing
if __name__ == "__main__":
    import http.server
    import socketserver
    from urllib.parse import urlparse, parse_qs
    
    class MetricsHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            
            # Enable CORS
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            
            try:
                if path == '/api/metrics/system':
                    metrics = metrics_api.get_system_metrics()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(metrics).encode())
                    
                elif path == '/api/metrics/agents':
                    agents = metrics_api.get_agent_metrics()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(agents).encode())
                    
                elif path.startswith('/api/metrics/agent/'):
                    agent_id = path.split('/')[-1]
                    agent_data = metrics_api.get_agent_metrics_by_id(agent_id)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(agent_data).encode())
                    
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'Not found')
                    
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f'Error: {str(e)}'.encode())
        
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
    
    # Start test server
    PORT = 8099
    with socketserver.TCPServer(("", PORT), MetricsHandler) as httpd:
        print(f"Metrics API server running on port {PORT}")
        print(f"Test endpoints:")
        print(f"  http://localhost:{PORT}/api/metrics/system")
        print(f"  http://localhost:{PORT}/api/metrics/agents")
        httpd.serve_forever() 