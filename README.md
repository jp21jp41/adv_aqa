# adv_aqa
Advanced Air Quality Analysis

Made with the help of Google Gemini, this project explores Scikit-Learn Machine Learning using Prompt/AI Engineering.

The project was initialized in early February 2026 as a showcase of more advanced machine learning tools, and it currently includes permutation importance, pipelines, encoders, column transformers, and error analysis, to name some.

However, the project has been reshaped into an AutoML project, and despite the advanced tools listed, the options are currently limited: there are column-transformed pipelines with the one type of feature engineering involved up to now (permutation importance) and without it. It does not have stronger ML tools yet.

There are multiple efficiency-based tradeoffs, as the degree of automation necessitates a larger codebase. One question worth considering: "Would a given amount of uncertainty be answered more quickly by automation, or by the developer(s)"?

Prompt Engineering provides strong boilerplate code, and with that, an easy way to reach basic AutoML code. That offers up the potential for code to be clean, efficient, and productive. It also means that further questions can be reached more quickly as exploration becomes both more direct and more expansive.
One example of conflicting ML decisions:
- With AI, one is able to write code that goes through many possible cases, meaning one could reach the answer by creating algorithms to uncover many different possible options from testing and going deeper into the testing based on the highest-scoring option(s). This option can go very slowly if the error improvement is too minimal (for example: testing every "main model" and getting a score of 55-65 out of 100, with very little improvement in deeper runs, may be rather time consuming to reach a more reasonable score).
- With AI, one can use mathematics to identify how the model should be analyzed. Statistics modules make these rather simple, and some mathematical conditions can be uncovered as simply as running a boolean condition applied to a statistics function result, especially when combined with AI. However, powerful use of mathematics in ML is generally more likely to need calculus and/or linear algebra at the minimum, as well as the ability to identify the problem and guide calculations in the proper direction; if it is with code itself, it helps to have functions understood as if they are mapped equations, and the same applies to AI, though the ability to specify, "Is it more efficient to have an optimization function ready to differentiate before or after the previous model score data winner is extracted?".

As the project continues, ML questions such as these will be explored further.

