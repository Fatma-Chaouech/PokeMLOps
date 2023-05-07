import sys
sys.path.append('src')
from opentelemetry import trace
from typing import List


def setup_telemetry():
    return trace.get_tracer(__name__)


def get_telemetry_args(parser):
    parser.add_argument('--experiment_name', type=str,
                        default='Pokemon Capturing', help='Name of the MLFlow experiment')
    return parser


def quick_span(tracer: object, span_name: str, log_name, log_values: List[float], info: str = None):
    with tracer.start_as_current_span(span_name):
        pass
