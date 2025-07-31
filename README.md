# Temporal_DECLARE_Constraints

Interpreting complex process executions remains a challenge for both analysts and domain experts. Declarative process models address this by capturing common behavioural patterns using general, rule-based constraints rather than detailing all execution paths. To improve explainability, we extend the Declare constraints in order to differentiate trace communities. Since standard declarative models lack the means to express when behaviours occur or how duration influences them, we introduce temporally enriched Declare constraints. These constraints incorporate time-related information, thus making the explanation framework more expressive and context-aware.
For each constraint within a community, we analyse the distribution of inter-activity durations to identify time intervals during which the constraint is frequently satisfied. This temporal refinement clarifies not only which constraints hold, but also when they typically hold, enabling both a global understanding of process variants and a local explanation of deviations between trace groups.

## Features

- **Community Discovery**: Identify behavioral communities in event logs
- **Constraint Mining**: Extract DECLARE constraints for each community
- **Temporal Analysis**: Analyze duration and time window patterns
- **Community Comparison**: Compare communities using Wasserstein distance
- **Visualization**: Generate heatmaps and explanatory visualizations




## Setup

- download repository
  ```bash
  git clone https://github.com/azinmoradbeikie/Temporal_DECLARE_Constraints
```bash
  pip install -r requirements.txt
