"""Run the four agents on the sample recordings, as if they had just arrived from the machine.

    python scripts/demo_pipeline.py
Prints each verdict and the advice, and writes the agent-to-agent message log to results/agent_log.jsonl.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from chatter_agents.orchestrator import default_coordinator  # noqa: E402


def alert(verdict):
    print(f"   ALERT  chatter on {verdict['source']} (P(stable) {verdict['p_stable']:.0%}) -> notify operator")


def main():
    coord = default_coordinator(alert_hook=alert)
    for out in coord.run_folder(REPO / "data" / "sample_recordings"):
        d, a = out["detection"], out["advice"]
        print(f"{d['source']:15} {d['rpm']:>7.0f} rpm {d['doc_mm']:>5.1f} mm -> {d['detected']:8}")
        print(f"   Agent 4: {a['recommendation']}")
    coord.save_log(REPO / "results" / "agent_log.jsonl")
    print(f"\n{len(coord.log)} agent messages -> results/agent_log.jsonl")


if __name__ == "__main__":
    main()
