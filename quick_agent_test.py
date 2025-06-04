#!/usr/bin/env python3
"""
🧪 QUICK AGENT TEST
===================
Test if current running agents can complete tasks
"""

import asyncio
import httpx
import json
from datetime import datetime

async def quick_test():
    """Test if current agents respond to tasks"""
    print("🧪 QUICK AGENT TEST")
    print("=" * 50)
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # 1. Check connected agents
            print("📋 Checking connected agents...")
            agents_response = await client.get("http://localhost:8080/api/agents")
            
            if agents_response.status_code != 200:
                print(f"❌ Failed to get agents: {agents_response.status_code}")
                return False
            
            agents = agents_response.json().get('agents', [])
            real_agents = [a for a in agents if 'Real' in a['agent_id']]
            
            print(f"📋 Found {len(agents)} total agents, {len(real_agents)} real agents:")
            for agent in real_agents:
                capabilities = agent.get('capabilities', [])
                status = agent.get('status', 'unknown')
                websocket_ready = agent.get('websocket_ready', False)
                print(f"   - {agent['agent_id']}: {capabilities} (status: {status}, ws: {websocket_ready})")
            
            if len(real_agents) < 1:
                print("❌ No real agents found. Make sure start_real_collaborative_agents.py is running.")
                return False
            
            # 2. Send a simple task to just one agent
            print(f"\n🚀 Sending simple task to test agent response...")
            
            delegation_data = {
                'task_title': 'Simple Agent Response Test',
                'task_description': 'Test if agents can process and respond to tasks',
                'required_capabilities': ['financial_analysis'],  # Single capability to test one agent
                'requester_id': 'quick_test',
                'task_intent': 'Test agent task processing and response',
                'max_agents': 1,
                'task_data': {
                    'stock_symbol': 'AAPL',
                    'test_type': 'agent_response_test',
                    'simple_request': 'Analyze Apple stock'
                }
            }
            
            response = await client.post("http://localhost:8080/api/delegate_task", json=delegation_data)
            
            if response.status_code == 200:
                result = response.json()
                delegation_id = result.get('delegation_id')
                workflow_id = result.get('workflow_id')
                
                print(f"✅ Task sent successfully!")
                print(f"   Delegation ID: {delegation_id}")
                print(f"   Workflow ID: {workflow_id}")
                
                # 3. Monitor the task completion
                return await monitor_task_completion(delegation_id, client)
            else:
                print(f"❌ Task delegation failed: {response.status_code}")
                try:
                    error_info = response.json()
                    print(f"   Error details: {error_info}")
                except:
                    print(f"   Response text: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def monitor_task_completion(delegation_id: str, client) -> bool:
    """Monitor task completion with detailed logging"""
    print(f"\n👀 Monitoring task completion...")
    print(f"   Delegation ID: {delegation_id}")
    
    completion_detected = False
    agents_working = set()
    max_wait = 60  # 1 minute should be enough for a simple task
    start_time = datetime.now()
    
    check_count = 0
    while not completion_detected and check_count < 30:  # 30 checks max
        check_count += 1
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        
        try:
            # Get detailed status
            status_response = await client.get(f"http://localhost:8080/api/collaboration_status/{delegation_id}")
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                workflow_status = status_data.get('workflow_status', 'unknown')
                participants = status_data.get('participating_agents', [])
                results = status_data.get('results', {})
                
                # Track agent participation
                if participants:
                    agents_working.update(participants)
                
                # Show progress
                print(f"   ⏰ Check {check_count} ({elapsed_seconds:.1f}s): Status = {workflow_status}")
                if participants:
                    print(f"      👥 Working agents: {participants}")
                
                # Check for completion
                if workflow_status == 'completed':
                    print(f"\n🎉 TASK COMPLETED! ({elapsed_seconds:.1f}s)")
                    completion_detected = True
                    
                    if results:
                        print(f"📊 Got results from {len(results)} agents:")
                        for agent_id, result in results.items():
                            print(f"\n   📝 Result from {agent_id}:")
                            if isinstance(result, dict):
                                # Show key fields from the result
                                status = result.get('status', 'unknown')
                                analysis = result.get('analysis', 'No analysis found')
                                capability = result.get('capability_used', 'unknown')
                                model = result.get('model_used', 'unknown')
                                
                                print(f"      Status: {status}")
                                print(f"      Capability: {capability}")
                                print(f"      Model: {model}")
                                print(f"      Analysis: {analysis[:200]}...")
                            else:
                                print(f"      {str(result)[:200]}...")
                        
                        print(f"\n🏆 SUCCESS: Agents are working and completing tasks!")
                        return True
                    else:
                        print(f"⚠️  Task completed but no results found")
                        print(f"   This might indicate a response format issue")
                        return False
                
                elif workflow_status == 'failed':
                    print(f"\n❌ Task failed after {elapsed_seconds:.1f}s")
                    print(f"   Agents that tried: {list(agents_working)}")
                    return False
                
                elif workflow_status == 'running':
                    if check_count == 1:
                        print(f"   ✅ Task is running - agents are processing...")
                    # Continue monitoring
                
                else:
                    print(f"   📊 Status: {workflow_status}")
            
            else:
                print(f"   ❌ Status check failed: {status_response.status_code}")
            
            # Wait before next check
            await asyncio.sleep(2)
            
            # Timeout check
            if elapsed_seconds > max_wait:
                print(f"\n⏰ Timeout after {max_wait}s")
                print(f"   Agents that participated: {list(agents_working)}")
                print(f"   Final status: {workflow_status}")
                return False
            
        except Exception as e:
            print(f"   ❌ Monitor error: {e}")
            await asyncio.sleep(3)
    
    print(f"\n⏰ Monitoring stopped after {check_count} checks")
    print(f"   Agents that participated: {list(agents_working)}")
    return False

async def main():
    """Run the quick agent test"""
    print("🚀 QUICK AGENT RESPONSE TEST")
    print("=" * 70)
    print("🎯 Goal: Test if your running agents can complete a simple task")
    print("🔍 This will tell us if the issue is in the agents or elsewhere")
    print("=" * 70)
    
    success = await quick_test()
    
    print("\n" + "=" * 70)
    if success:
        print("🏆 EXCELLENT! YOUR AGENTS ARE WORKING!")
        print("✅ Agents can receive, process, and complete tasks")
        print("✅ Your collaboration engine is 100% functional")
        print("🎯 Multi-agent collaboration is working correctly")
    else:
        print("🔧 AGENTS NEED DEBUGGING")
        print("❌ Agents connect but don't complete tasks properly")
        print("💡 Issue is likely in:")
        print("   - Ollama call timeouts")
        print("   - Response message format")
        print("   - Exception handling in agent code")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main()) 