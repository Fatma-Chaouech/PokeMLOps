from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleExportSpanProcessor

from mlflow.tracking import MlflowClient
import mlflow

def open_telemetry():
    # configure the OpenTelemetry exporter
    otlp_exporter = OTLPSpanExporter(endpoint="localhost:55680")
    span_processor = SimpleExportSpanProcessor(otlp_exporter)
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(span_processor)

    # initialize the MLFlow client
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("my_experiment")

    # create a new tracing client that uses the MLFlow client and the OpenTelemetry tracer provider
    mlflow_client = MlflowClient()
    tracer = trace_provider.get_tracer(__name__)
    mlflow_client._tracking_client.tracking_service._client.add_span_processor(
        trace_provider.get_span_processor()
    )

    return tracer