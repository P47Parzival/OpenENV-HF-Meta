"""
inference.py - Inference Client for Support Ticket Triage

This script provides a clean, user-friendly client to connect to the
OpenEnv Support Ticket Triage environment (whether local or on Hugging Face Spaces).
It includes an interactive manual mode as well as an LLM automated mode.

Usage:
  # Manual interaction (human plays the agent)
  python inference.py --manual --base-url http://localhost:7860

  # Automated LLM inference using Gemini
  export GEMINI_API_KEY="your-key"
  python inference.py --base-url http://localhost:7860
"""

import argparse
import json
import os
import sys

import httpx


class SupportEnvClient:
    """A Python client for interacting with the Support Ticket Triage environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def is_healthy(self) -> bool:
        """Check if the environment server is reachable."""
        try:
            resp = self.client.get("/health")
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    def get_tasks(self) -> dict:
        """Fetch all available tasks."""
        return self.client.get("/tasks").json()

    def reset(self, task_idx: int = 0) -> dict:
        """Start a new episode with the specified task."""
        return self.client.post(f"/reset?task_idx={task_idx}").json()

    def step(self, ticket_id: str, action_type: str, target: str = None, message: str = None) -> dict:
        """Submit an action for a specific ticket."""
        payload = {
            "ticket_id": ticket_id,
            "action_type": action_type
        }
        if target:
            payload["target"] = target
        if message:
            payload["message"] = message

        resp = self.client.post("/step", json=payload)
        if resp.status_code != 200:
            raise ValueError(f"Step failed: {resp.text}")
        return resp.json()

    def fetch_grader_score(self) -> dict:
        """Fetch the final graded score of the episode."""
        return self.client.get("/grader").json()

    def close(self):
        self.client.close()


def manual_inference(client: SupportEnvClient):
    """Interactive loop to manually test the environment."""
    print("Welcome to Support Ticket Triage - Manual Mode")
    tasks_info = client.get_tasks()
    print("\nAvailable Tasks:")
    for t in tasks_info["tasks"]:
        print(f"  [{t['id']}] {t['difficulty']} - {t['description']}")
    
    try:
        task_idx = int(input("\nSelect a task index (0, 1, or 2): "))
    except ValueError:
        task_idx = 0

    print(f"\nResetting environment for task {task_idx}...")
    obs = client.reset(task_idx)

    # Inference Loop
    done = False
    while not done:
        open_tickets = obs.get("open_tickets", [])
        if not open_tickets:
            print("No more open tickets.")
            break
            
        print("\n" + "="*50)
        print(f"STEP: {obs.get('step_number', 0)} / {obs.get('max_steps', '?')}")
        print("OPEN TICKETS:")
        for t in open_tickets:
            print(f"\n  ID: {t['id']} | Status: {t['status']} | Category: {t.get('category')} | Route: {t.get('assigned_to')}")
            print(f"  Subject: {t['subject']}")
            print(f"  Body: {t['body']}")
        print("="*50)

        print("\nProvide an action. Valid types: classify, prioritize, route, respond, escalate, resolve")
        ticket_id = input("Ticket ID to act on (or 'q' to quit): ").strip()
        if ticket_id.lower() == 'q':
            break
            
        action_type = input("Action Type: ").strip()
        target = input("Target (optional, press Enter to skip): ").strip()
        target = target if target else None
        
        message = input("Message (optional, press Enter to skip): ").strip()
        message = message if message else None

        try:
            result = client.step(ticket_id, action_type, target, message)
            obs = result["observation"]
            done = result["done"]
            print(f"\n✅ Action Processed! Reward earned: {result['reward']:.2f}")
            print(f"📝 Feedback: {result['info'].get('feedback', '')}")
        except Exception as e:
            print(f"\n❌ Error submitting action: {e}")

    # Display final score
    grader = client.fetch_grader_score()
    print("\n" + "*"*50)
    print("EPISODE FINISHED")
    print(f"Final Score: {grader['score']:.4f}")
    print(f"Max Possible Reward: {grader['max_possible_reward']:.4f}")
    print("*"*50)


def automated_inference(client: SupportEnvClient):
    """Run an automated LLM inference loop using Gemini."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("Please 'pip install google-generativeai' for automated mode, or use --manual")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable not set. Please set it or run with --manual.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    print("\nStarting automated inference loop over all 3 tasks...")
    
    for task_idx in range(3):
        print(f"\n--- Running Task {task_idx} ---")
        obs = client.reset(task_idx)
        done = False
        
        while not done:
            open_tickets = obs.get("open_tickets", [])
            if not open_tickets:
                break
                
            prompt = f"You are processing customer support tickets. Open tickets:\n"
            for t in open_tickets:
                prompt += f"ID: {t['id']} | Subject: {t['subject']} | Body: {t['body']}\n"
            
            prompt += """
Produce exactly ONE action in JSON format for the highest priority ticket.
Schema: {"ticket_id": "T001", "action_type": "classify", "target": "it_support"}
Action types: classify, prioritize, route, respond, escalate, resolve.
Return ONLY valid JSON.
"""
            # Ask Gemini for the next action step
            response = model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            
            try:
                action_data = json.loads(text)
                ticket_id = action_data.get("ticket_id")
                action_type = action_data.get("action_type")
                target = action_data.get("target")
                message = action_data.get("message")
                
                print(f"Robot decides: {action_type} on ticket {ticket_id} (target: {target})")
                
                result = client.step(ticket_id, action_type, target, message)
                obs = result["observation"]
                done = result["done"]
                print(f"Environment returned: Reward {result['reward']:.2f}")
                
            except json.JSONDecodeError:
                print(f"Model returned invalid JSON: {text}. Ending episode early.")
                break
            except Exception as e:
                print(f"Error executing step: {e}. Ending episode early.")
                break
                
        # Final Score
        grader = client.fetch_grader_score()
        print(f"Task {task_idx} Complete! Normalized Score: {grader['score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Support Ticket Triage Inference Client")
    parser.add_argument("--base-url", default="http://localhost:7860", help="URL of the OpenEnv Server")
    parser.add_argument("--manual", action="store_true", help="Run in interactive manual mode")
    
    args = parser.parse_args()
    
    client = SupportEnvClient(args.base_url)
    
    print(f"Attempting to connect to environment server at: {args.base_url}")
    if not client.is_healthy():
        print(f"❌ Server at {args.base_url} is not responding or unhealthy.")
        print("Please make sure the environment is running ('python main.py').")
        sys.exit(1)
        
    print("✅ Connected to environment.")

    try:
        if args.manual:
            manual_inference(client)
        else:
            automated_inference(client)
    except KeyboardInterrupt:
        print("\nInference interrupted. Exiting.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
