import json
from datetime import datetime
from pathlib import Path

from .models import Agent

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "agents"


class AgentManager:
    def __init__(self):
        BASE_DIR.mkdir(parents=True, exist_ok=True)

    def create_agent(self, name: str, description: str, system_prompt: str, **kwargs) -> Agent:
        agent = Agent(
            name=name,
            description=description,
            system_prompt=system_prompt,
            **kwargs,
        )
        self._save_agent(agent)
        return agent

    def get_agent(self, agent_id: str) -> Agent | None:
        try:
            return self._load_agent(agent_id)
        except FileNotFoundError:
            return None

    def list_agents(self) -> list[Agent]:
        agents = []
        for file in BASE_DIR.glob("*.json"):
            data = json.loads(file.read_text(encoding="utf-8"))
            agents.append(Agent(**data))
        return agents

    def update_agent(self, agent_id: str, **kwargs) -> Agent:
        agent = self._load_agent(agent_id)
        updated_data = agent.model_dump()
        updated_data.update(kwargs)
        updated_data["updated_at"] = datetime.now()
        agent = Agent(**updated_data)
        self._save_agent(agent)
        return agent

    def delete_agent(self, agent_id: str) -> bool:
        path = BASE_DIR / f"{agent_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def _save_agent(self, agent: Agent) -> None:
        path = BASE_DIR / f"{agent.id}.json"
        path.write_text(agent.model_dump_json(indent=2), encoding="utf-8")

    def _load_agent(self, agent_id: str) -> Agent:
        path = BASE_DIR / f"{agent_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return Agent(**data)
