from messaging import *
from agent.base_agent import Agent
from enum import Enum

class AgentType(str, Enum):
    VISION = 'Vision'
    PLANNING = 'Planning'
    EXECUTION = 'Execution'
    AUDIO = 'Audio'

class Coordinator:
    def __init__(self):
        self.all_agents: dict[str, Agent] = {}
    def add_agent(self, agent: Agent):
        self.all_agents[agent.name] = agent

    def dispatch(self, message: Message)->Message:
        receiver = message.receiver
        if receiver not in self.all_agents:
            return Message(
                sender="coordinator",
                receiver=message.sender,
                action=f"{message.action}",
                payload={},
                message_type=MessageType.RESPONSE,
                status=MessageStatus.ERROR,
                parent_id=message.parent_id,
                error=f"Agent '{receiver}' is not registered with the Coordinator.",
            )
        target = self.all_agents[receiver]
        try:
            return target.handle_message(message)
        except Exception as e:
            return Message(
                sender="coordinator",
                receiver=message.sender,
                action=f"{message.action}",
                payload={},
                message_type=MessageType.RESPONSE,
                status=MessageStatus.ERROR,
                parent_id=message.parent_id,
                error=f"Agent '{receiver}' crashed while processing: {str(e)}",
            )