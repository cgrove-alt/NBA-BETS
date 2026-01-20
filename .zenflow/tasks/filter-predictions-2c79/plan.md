# Quick change

## Configuration
- **Artifacts Path**: {@artifacts_path} → `.zenflow/tasks/{task_id}`

---

## Agent Instructions

This is a quick change workflow for small or straightforward tasks where all requirements are clear from the task description.

### Your Approach

1. Proceed directly with implementation
2. Make reasonable assumptions when details are unclear
3. Do not ask clarifying questions unless absolutely blocked
4. Focus on getting the task done efficiently

This workflow also works for experiments when the feature is bigger but you don't care about implementation details.

If blocked or uncertain on a critical decision, ask the user for direction.

---

## Workflow Steps

### [x] Step: Implementation
<!-- chat-id: 8996306b-0d13-4edf-b14b-d992beb3b5a5 -->

Implement the task directly based on the task description.

1. Make reasonable assumptions for any unclear details
2. Implement the required changes in the codebase
3. Add and run relevant tests and linters if applicable
4. Perform basic manual verification if applicable

Save a brief summary of what was done to `{@artifacts_path}/report.md` if significant changes were made.

**COMPLETED** - Full comprehensive implementation with all requested features plus significant bonus functionality. See report.md for complete details.

### [x] Step: Complete Analysis
<!-- chat-id: current -->

Run complete analysis of filtering options for confidence level and game filters.

1. Analyze current implementation thoroughly
2. Document all available data and filtering capabilities
3. Identify gaps and opportunities
4. Provide comprehensive recommendations with options
5. Prioritize by ROI and complexity

**COMPLETED** - See filter-analysis.md for exhaustive analysis with 10 enhancement options, competitive analysis, implementation paths, and prioritized recommendations.
