# M52.1 topology hotfix

Fixes Bash `set -u` failures caused by using `tag` inside the same `local` assignment where it is first defined.

For an interrupted M52 run that already reached the one-time prefault stage, run:

```bash
PROJECT=~/asm/diffsinger bash run_m52_resume_fixed.sh
```

For a clean full rerun:

```bash
PROJECT=~/asm/diffsinger bash run_m52_topology_allinone_fixed.sh
```
