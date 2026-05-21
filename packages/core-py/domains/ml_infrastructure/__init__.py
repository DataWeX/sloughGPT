"""
ML Infrastructure - Production-Grade ML Engineering Components

This module provides enterprise-grade ML infrastructure components:

1. Experiment Tracker - Track ML experiments, parameters, metrics, artifacts
2. Model Versioning - Version control for ML models with stages and lineage
3. Feature Store - Feature engineering, storage, and serving
4. Hyperparameter Tuner - Automated hyperparameter optimization
5. Data Pipeline - ETL and data processing for ML
6. Evaluation - Metrics and evaluation utilities
7. Callbacks - Training hooks and callbacks
8. Model Serving - Production model deployment and inference
9. Model Monitoring - Drift detection and performance monitoring

Usage:
    from domains.ml_infrastructure import (
        experiment_tracker,
        model_versioning,
        feature_store,
        hyperparameter_tuner,
        data_pipeline,
        evaluation,
        callbacks,
        model_serving,
        model_monitoring
    )
        evaluation,
        callbacks
    )

    # Or import specific components
    from domains.ml_infrastructure.experiment_tracker import start_run, log_metric
    from domains.ml_infrastructure.model_versioning import register_model, ModelStage
    from domains.ml_infrastructure.feature_store import register_feature, FeatureType
    from domains.ml_infrastructure.hyperparameter_tuner import create_tuning_job, SearchStrategy
    from domains.ml_infrastructure.data_pipeline import DataPipeline, from_file
    from domains.ml_infrastructure.evaluation import MetricsCalculator, ClassificationMetrics
    from domains.ml_infrastructure.callbacks import EarlyStoppingCallback, ModelCheckpointCallback
"""

from contextlib import contextmanager


from domains.ml_infrastructure.experiment_tracker import (
    ExperimentTracker,
    Experiment,
    MetricPoint,
    ExperimentStatus,
    tracker,
    create_experiment,
    start_run,
    log_parameter,
    log_parameters,
    log_metric,
    log_metrics,
    log_artifact,
)

from domains.ml_infrastructure.model_versioning import (
    ModelVersioning,
    Model,
    ModelVersion,
    ModelMetrics,
    ModelStage,
    ModelStatus,
    model_registry,
    register_model,
    create_model_version,
    get_model,
    list_models,
)

from domains.ml_infrastructure.feature_store import (
    FeatureStore,
    Feature,
    FeatureGroup,
    FeatureVector,
    FeatureStats,
    FeatureType,
    FeatureStatus,
    FeatureValidator,
    feature_store,
    register_feature,
    register_feature_group,
    get_feature,
    get_feature_vector,
)

from domains.ml_infrastructure.hyperparameter_tuner import (
    HyperparameterTuner,
    TuningJob,
    Trial,
    HyperparameterSpec,
    HyperparameterSearchSpace,
    SearchSpace,
    TuneStatus,
    TuneObjective,
    SearchStrategy,
    BaseOptimizer,
    RandomOptimizer,
    GridOptimizer,
    BayesianOptimizer,
    tuner,
    create_tuning_job,
    start_tuning,
    get_best_params,
)

from domains.ml_infrastructure.data_pipeline import (
    DataPipeline,
    DataSource,
    FileSource,
    InMemorySource,
    DataSchema,
    DataStats,
    DataType,
    PipelineStatus,
    Transform,
    FilterTransform,
    MapTransform,
    RenameTransform,
    DropTransform,
    FillNaTransform,
    CompositeTransform,
    FilteredOut,
    pipeline,
    from_file,
    from_list,
)

from domains.ml_infrastructure.evaluation import (
    MetricsCalculator,
    ClassificationMetrics,
    RegressionMetrics,
    NLPMetrics,
    CrossValidator,
    ModelComparator,
    calculator,
)

from domains.ml_infrastructure.callbacks import (
    Callback,
    CallbackOrder,
    TrainPhase,
    TrainerState,
    CallbackList,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LearningRateSchedulerCallback,
    LoggingCallback,
    TensorBoardCallback,
)

from domains.ml_infrastructure.model_monitoring import (
    ModelMonitor,
    DriftDetector,
    MetricsAggregator,
    AlertManager,
    DriftResult,
    Alert,
    MetricValue,
    MonitorStatus,
    monitor,
)


__all__ = [
    # Experiment Tracker
    "ExperimentTracker",
    "Experiment",
    "MetricPoint",
    "ExperimentStatus",
    "tracker",
    "create_experiment",
    "start_run",
    "log_parameter",
    "log_parameters",
    "log_metric",
    "log_metrics",
    "log_artifact",
    # Model Versioning
    "ModelVersioning",
    "Model",
    "ModelVersion",
    "ModelMetrics",
    "ModelStage",
    "ModelStatus",
    "model_registry",
    "register_model",
    "create_model_version",
    "get_model",
    "list_models",
    # Feature Store
    "FeatureStore",
    "Feature",
    "FeatureGroup",
    "FeatureVector",
    "FeatureStats",
    "FeatureType",
    "FeatureStatus",
    "FeatureValidator",
    "feature_store",
    "register_feature",
    "register_feature_group",
    "get_feature",
    "get_feature_vector",
    # Hyperparameter Tuner
    "HyperparameterTuner",
    "TuningJob",
    "Trial",
    "HyperparameterSpec",
    "HyperparameterSearchSpace",
    "SearchSpace",
    "TuneStatus",
    "TuneObjective",
    "SearchStrategy",
    "BaseOptimizer",
    "RandomOptimizer",
    "GridOptimizer",
    "BayesianOptimizer",
    "tuner",
    "create_tuning_job",
    "start_tuning",
    "get_best_params",
    # Data Pipeline
    "DataPipeline",
    "DataSource",
    "FileSource",
    "InMemorySource",
    "DataSchema",
    "DataStats",
    "DataType",
    "PipelineStatus",
    "Transform",
    "FilterTransform",
    "MapTransform",
    "RenameTransform",
    "DropTransform",
    "FillNaTransform",
    "CompositeTransform",
    "FilteredOut",
    "pipeline",
    "from_file",
    "from_list",
    # Evaluation
    "MetricsCalculator",
    "ClassificationMetrics",
    "RegressionMetrics",
    "NLPMetrics",
    "CrossValidator",
    "ModelComparator",
    "calculator",
    # Callbacks
    "Callback",
    "CallbackOrder",
    "TrainPhase",
    "TrainerState",
    "CallbackList",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "LearningRateSchedulerCallback",
    "LoggingCallback",
    "TensorBoardCallback",
    # Model Serving
    "ModelServer",
    "ModelEndpoint",
    "InferenceEngine",
    "InferenceRequest",
    "InferenceResponse",
    "BatchingManager",
    "ModelRouter",
    "ModelStatus",
    "server",
    # Model Monitoring
    "ModelMonitor",
    "DriftDetector",
    "MetricsAggregator",
    "AlertManager",
    "DriftResult",
    "Alert",
    "MetricValue",
    "MonitorStatus",
    "monitor",
]
