# adv_aqa
Advanced Air Quality Analysis

Made with the help of Google Gemini, this project explores Scikit-Learn Machine Learning using Prompt/AI Engineering.

The project was initialized on January 20, 2026 as a showcase of more advanced machine learning tools, and it currently includes permutation importance, pipelines, encoders, column transformers, and error analysis, to name some.

However, the project has been reshaped into an AutoML project, and despite the advanced tools listed, the options are currently limited: there are column-transformed pipelines with the one type of feature engineering involved up to now (permutation importance) and without it. It does not have stronger ML tools yet.

There are multiple efficiency-based tradeoffs, as the degree of automation necessitates a larger codebase. One question worth considering: "Would a given amount of uncertainty be answered more quickly by automation, or by the developer(s)"?

Prompt Engineering provides strong boilerplate code, and with that, an easy way to reach basic AutoML code. That offers up the potential for code to be clean, efficient, and productive. It also means that further questions can be reached more quickly as exploration becomes both more direct and more expansive.
One example of conflicting ML decisions:
- With AI, one is able to write code that goes through many possible cases, meaning one could reach the answer by creating algorithms to uncover many different possible options from testing and going deeper into the testing based on the highest-scoring option(s). This option can go very slowly if the error improvement is too minimal (for example: testing every "main model" and getting a score of 55-65 out of 100, with very little improvement in deeper runs, may be rather time consuming to reach a more reasonable score).
- With AI, one can use mathematics to identify how the model should be analyzed. Statistics modules make these rather simple, and some mathematical conditions can be uncovered as simply as running a boolean condition applied to a statistics function result, especially when combined with AI. However, powerful use of mathematics in ML is generally more likely to need calculus and/or linear algebra at the minimum, as well as the ability to identify the problem and guide calculations in the proper direction; if it is with code itself, it helps to have functions understood as if they are mapped equations, and the same applies to AI, though the ability to specify, "Is it more efficient to have an optimization function ready to differentiate before or after the previous model score data winner is extracted?".

As the project continues, ML questions such as these will be explored further.


# 3/9/2026 Update (Loose Evaluation of Google Gemini/Google Copilot Search and Revealing Unknowns):
The project is making significant progress, though not without flaw. 

Machine Learning is not inherently supposed to solve for the "unknown". However, because Machine Learning arguably has the most benefit when the information extracted from data is maximized, and for the purpose of advancing our knowledge of Machine Learning, more unknowns may be welcome. In an attempt to maximize its benefits, the project had at first been run trying to avoid receiving knowledge about the dataset from outside sources. The data of course already has algorithms involved in its processes, but they were avoided before Google Gemini offered an equation for AQI based in part on the Arithmetic Mean column. The idea had not particularly affected how the model was assessed, but it was a clear sign that, with data that is already rather expansive in how it is known to be, pushing in the direction of them gives room towards further inspection.

While Google Gemini's accidental discovery forced progress, that equation was still _initially unwanted_. The data is not expected to be calculated with the equation, especially when:
i. The data is partially biased by certain other variables regardless of the equation out of context.
ii. Other metrics beyond the AQI as a predicted variable have not been considered yet.

Of course, Google Copilot Search was mentioned. Google Gemini is definitely more reliable for answering questions dynamically, but the Copilot Search has helped as a second source of information when Google Gemini seems to struggle.

