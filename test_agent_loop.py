from backend.agent_loop import AgentLoop

id_map = {1: (100, 200), 2: (300, 400), 5: (500, 600)}

# test click
action = {"command": "click", "target_id": 1, "reason": "Test click"}
print(AgentLoop._parse_action(action, id_map))

# test drag
action = {"command": "drag", "source_id": 2, "target_id": 5, "reason": "Test drag"}
print(AgentLoop._parse_action(action, id_map))

# test scroll
action = {"command": "scroll", "target_id": 1, "amount": 10, "reason": "Test scroll"}
print(AgentLoop._parse_action(action, id_map))

# fallback string
print(AgentLoop._parse_action("press enter"))
