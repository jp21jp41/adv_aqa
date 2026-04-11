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

# 4/1/2026 Update (Random Forest over Linear Models)
Google Gemini was able to handle the idea of the different models involved in improving the SMAPE score. The issue of grouping data surfaced, and in order to keep going through with that to improve accuracy, the VIF was analyzed. The main problem seemed to be that some of the larger State variables (which are places such as California and Hawaii) had a lot of variability, while there was still multicollinearity amongst several states. There were many different possible tools (not mutually exclusive, but not necessarily reasonable to combine):

- State AQI Mean
- PCA Spatial Transformer
- Random Forest Regressor

The State-based mean was the easier Linear Model Solution, but it likely confounds with both of the other tools, linearly or not. Regardless, the goal was to make another great jump in accuracy after having a strong improvement through the duration correction through adjusting the Sample Duration based on the data (thankfully, it was designed such that duration-based groups could simply be summed together to record a sample duration) combined with imputation.

The Random Forest Regressor improved the SMAPE Score from about 36.5% to about 27%, whereas the other tools didn't make much difference. One other option is changing from OneHot to Target Encoding. Right now, with the State AQI Mean, TargetEncoding _seems_ to be more accurate, but that may change. The Random Forest Regressor should be run separately from the other tools mentioned, but it hasn't yet because the runtime limitations had been increasing, with a more construction-focused headway to be noted.

#########################

A custom path to a separate Scikit-Learn module has been included in the system. The idea is that once the model has reached a certain level of accuracy, the separate Scikit-Learn module would be patched to run faster. While algorithmic efficiency can be better within the Air Quality Analysis module, the system efficiency may not have such an advantage because of the connection that Scikit-Learn has to certain components such as Cython, NumPy, SciPy, BLAS/LAPACK, and Joblib. One would not expect in this current case to be able to unpack everything so readily, but the connection to a separate module at least makes one testing form of Scikit-Learn with a baseline one simpler. Even if some pure algorithms replace certain Scikit-Learn functions, it would likely be more difficult for more extensive functions and much of the improvements might expectantly be from simply removing exogenous functions in these modules.

The custom path was quickly included with the help of Google Gemini. That was not the only improvement. Of course, runtime efficiency is to be observed with a time cache, and that was added with the help of Google Gemini, which was much more concise than making machine learning improvements with it. Google Gemini will direct a developer to where new code has to placed, but moreso when explicitly told to; that much is assumable given that it tends to perform better with immediate ideas (meaning ideas that are typed into more recent prompts). However, Google Gemini is fantastic at giving the user immediate tools that wouldn't be readily available without it. Google Gemini made, to solve the problem of needing run ID's in the cache, a backfill module and then a reordering module, solving a potential whole other undertaking in about 10 minutes, with each module performing their respective tasks instantaneously.

Google Gemini metaphorically offers a developer a guided toolbox for every tool needed, which is a great advantage.

NOTE: The goal is not to dismantle Google Gemini. It is a great, efficient system that has helped me exceptionally through this process.

# 4/2/2026 Update (Search CV)
A RandomSearchCV has already been run on the data. Unfortunately, this is where lengthy runtimes really begin to take a toll. The runs with Random Forests were already beginning to slow down to about 20 minutes, but those were still reasonable given a limited number of runs. The RandomSearchCV has only been given 1 proper run, which took over an hour, and given the fact that the new version in the making doesn't support hyperparameters, a time cache entry wasn't recorded at the time.
This all would imply that runs require sampling of the data set, and of course, it would be the purpose for a custom, vendor-patched machine learning framework. If the framework were fast enough, running the module with the Search CV would be reasonable.

# 4/11/2026 Update (Potential New Metrics/Cache Edits, Hyperparameter Tuning and Considerations for Custom Scikit-Learn Usage)
Up to now, the SMAPE has mostly been relied on for calculating accuracy. Although, at this time, the potential for using the MAE as well as the Negative Mean Poisson Deviance has been posed by AI. While those different metrics may not be necessary, they are worth considering, and with all of the other fixes, the project is at crossroads with several potential tasks:
- The New Metrics
- Making a cache system that properly includes certain metrics, as well as, given the limitations of a large cache, the ability to cut down unnecessary cache data
- The "Directional Shift" of Hyperparameters (according to Google Gemini)
- Adding other potential steps to the machine learning model if necessary
- Designing what may possibly be multiple custom Scikit-Learn packages to meet the different needs of each system, as well as an overencompassing one
- Potentially, practical cases that extract data that use one or more of the models
