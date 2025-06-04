#!/usr/bin/env python3
"""
🎯 TEST REAL AGENTS SPECIFICALLY
================================
Target our Real agents specifically to see if they respond
"""

import asyncio
import httpx
import json
from datetime import datetime

async def test_real_agents():
    """Test real agents specifically"""
    print("🎯 TESTING REAL AGENTS SPECIFICALLY")
    print("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # 1. Check Real agents specifically
            print("📋 Checking Real agents...")
            agents_response = await client.get("http://localhost:8080/api/agents")
            
            if agents_response.status_code != 200:
                print(f"❌ Failed to get agents: {agents_response.status_code}")
                return False
            
            agents = agents_response.json().get('agents', [])
            real_agents = [a for a in agents if a['agent_id'].startswith('Real')]
            
            print(f"📋 Found {len(real_agents)} Real agents:")
            for agent in real_agents:
                capabilities = agent.get('capabilities', [])
                status = agent.get('status', 'unknown')
                websocket_ready = agent.get('websocket_ready', False)
                print(f"   - {agent['agent_id']}: {capabilities}")
                print(f"     Status: {status}, WebSocket: {websocket_ready}")
            
            if len(real_agents) < 1:
                print("❌ No Real agents found!")
                return False
            
            # 2. Send task with capabilities ONLY Real agents have
            print(f"\n🚀 Sending task targeting Real agents...")
            
            # Use capabilities that ONLY our Real agents should have
            delegation_data = {
                'task_title': 'Real Agent Specific Test',
                'task_description': 'Test targeting Real agents only',
                'required_capabilities': [
                    'market_analysis',      # RealFinancialAnalyst_001 specific
                    'data_visualization'    # RealContentCreator_002 specific  
                ],
                'requester_id': 'real_agent_test',
                'task_intent': 'Target Real agents with their unique capabilities',
                'max_agents': 2,
                'task_data': {
                    'stock_symbol': 'TSLA',
                    'real_agent_test': True,
                    'specific_request': 'This should go to Real agents only'
                }
            }
            
            response = await client.post("http://localhost:8080/api/delegate_task", json=delegation_data)
            
            if response.status_code == 200:
                result = response.json()
                delegation_id = result.get('delegation_id')
                
                print(f"✅ Task sent: {delegation_id}")
                
                # 3. Monitor specifically for Real agents
                return await monitor_real_agents(delegation_id, client)
            else:
                print(f"❌ Task delegation failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def monitor_real_agents(delegation_id: str, client) -> bool:
    """Monitor specifically for Real agent responses"""
    print(f"\n👀 Monitoring for Real agent responses...")
    
    real_agents_working = set()
    max_wait = 45
    start_time = datetime.now()
    
    for check in range(25):  # 25 checks over ~50 seconds
        elapsed = (datetime.now() - start_time).total_seconds()
        
        try:
            status_response = await client.get(f"http://localhost:8080/api/collaboration_status/{delegation_id}")
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                workflow_status = status_data.get('workflow_status', 'unknown')
                participants = status_data.get('participating_agents', [])
                results = status_data.get('results', {})
                
                # Track Real agents specifically
                real_participants = [p for p in participants if p.startswith('Real')]
                if real_participants:
                    real_agents_working.update(real_participants)
                
                print(f"   ⏰ Check {check+1} ({elapsed:.1f}s): {workflow_status}")
                
                if participants:
                    print(f"      👥 All working: {participants}")
                if real_participants:
                    print(f"      🎯 Real agents: {real_participants}")
                
                # Success conditions
                if workflow_status == 'completed':
                    print(f"\n🎉 TASK COMPLETED! ({elapsed:.1f}s)")
                    
                    real_results = {k: v for k, v in results.items() if k.startswith('Real')}
                    
                    if real_results:
                        print(f"🏆 SUCCESS: Real agents completed tasks!")
                        print(f"   Real agents that responded: {list(real_results.keys())}")
                        
                        for agent_id, result in real_results.items():
                            print(f"\n   📝 {agent_id} result:")
                            if isinstance(result, dict):
                                status = result.get('status', 'unknown')
                                print(f"      Status: {status}")
                                if 'analysis' in result:
                                    analysis = str(result['analysis'])[:150]
                                    print(f"      Analysis: {analysis}...")
                                elif 'result' in result:
                                    result_text = str(result['result'])[:150]
                                    print(f"      Result: {result_text}...")
                        return True
                    else:
                        print(f"⚠️  Task completed but no Real agent results")
                        print(f"   All results: {list(results.keys())}")
                        return False
                
                elif workflow_status == 'failed':
                    print(f"\n❌ Task failed")
                    return False
                
                elif elapsed > max_wait:
                    print(f"\n⏰ Timeout after {max_wait}s")
                    print(f"   Real agents that tried: {list(real_agents_working)}")
                    return False
            
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Monitor error: {e}")
            await asyncio.sleep(3)
    
    print(f"\n⏰ Monitoring complete")
    print(f"   Real agents that participated: {list(real_agents_working)}")
    return len(real_agents_working) > 0

async def main():
    """Run Real agent test"""
    print("🚀 REAL AGENT SPECIFIC TEST")
    print("=" * 70)
    print("🎯 Target: Test Real agents with their unique capabilities")  
    print("🔍 Purpose: See if Real agents respond or if others are selected")
    print("=" * 70)
    
    success = await test_real_agents()
    
    print("\n" + "=" * 70)
    if success:
        print("🏆 REAL AGENTS ARE WORKING!")
        print("✅ Your Real agents can complete tasks")
        print("🎯 Collaboration system is functional")
    else:
        print("🔧 REAL AGENT ISSUE IDENTIFIED")
        print("💡 Either:")
        print("   - Real agents aren't being selected (capability mismatch)")
        print("   - Real agents are selected but don't respond (code bug)")
        print("   - Other agents are being chosen instead")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main()) 