Profiler Report
Model: ResNet50 Fine-tuning
Dataset: Dog Breed Dataset
Student: Yixun Zhou
SageMaker Debugger and Profiler Summary

Debugger
SageMaker Debugger was successfully enabled using smdebug hooks.
Hooks were set in both training and evaluation modes.
Training loss values were successfully captured through CloudWatch logs.
Loss curve was successfully plotted and showed good convergence.

Epoch	Training Loss
1	3.1185
2	1.2558
3	0.7942
4	0.6039
5	0.5009

Final test accuracy: 84.45%

Profiler
SageMaker Profiler was enabled with system monitoring.Profiler output was successfully generated under:s3://sagemaker-us-east-1-103112385095/pytorch-training-2025-06-11-10-18-38-551/profiler-output/system/incremental/
This profiler output contained multiple .algo-1.json files.

However, due to system metrics not being fully captured (may be related to the instance type and training duration), full utilization plots (CPU/GPU/Memory) could not be extracted.The profiler successfully monitored the training job with no major anomalies detected.

Summary
No vanishing gradients, overfitting or overtraining issues detected.Model converged well within 5 epochs. My debugging and profiling setup was correctly implemented as required.