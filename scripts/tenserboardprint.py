from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tb_dir = Path(
    "/opt/exps/ReSeek/tensorboard_log/"
    "ReSeek/"
    "ReSeek-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo_max_turn1"
)

event_files = list(tb_dir.rglob("events.out.tfevents.*"))

if not event_files:
    raise FileNotFoundError(f"没有找到 TensorBoard event 文件：{tb_dir}")

event_files.sort(key=lambda p: p.stat().st_mtime)
latest_event = event_files[-1]

print(f"共找到 {len(event_files)} 个 event 文件")
print(f"本次只读取最新文件：{latest_event}")
print(f"修改时间：{latest_event.stat().st_mtime}")
print()

ea = EventAccumulator(
    str(latest_event),
    size_guidance={"scalars": 0},
)
ea.Reload()

scalar_tags = sorted(ea.Tags().get("scalars", []))

for tag in scalar_tags:
    if not tag.startswith(("val-core/", "val-aux/")):
        continue

    events = ea.Scalars(tag)
    if not events:
        continue

    latest = events[-1]
    print(f"{tag}: {latest.value:.6f}  step={latest.step}")