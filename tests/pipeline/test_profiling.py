"""Synthetic profiling checks without loading models or reading pipeline data."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
import pathlib
import sys
import unittest


spec = importlib.util.spec_from_file_location('stage_profiling_test_module', pathlib.Path(__file__).parents[2] / 'pii_pipeline/profiling.py')


p = importlib.util.module_from_spec(spec)


sys.modules[spec.name] = p


spec.loader.exec_module(p)


class ProfilingTests(unittest.TestCase):
    def test_cpu_accounting(self):
        profiler = p.PipelineProfiler()
        with patch.object(p.time, 'perf_counter', side_effect=[10., 12.]), patch.object(p.os, 'times', side_effect=[SimpleNamespace(user=1., system=2.), SimpleNamespace(user=4., system=3.)]):
            with p.profile_context('cpu', profiler=profiler):
                pass
        stage = profiler.stages[0]
        self.assertEqual(stage.cpu_percent, 200.)
        self.assertEqual(stage.cpu_cores_used, 2.)
        self.assertEqual(stage.duration_ms, 2000.)

    def test_nesting_and_exception(self):
        profiler = p.PipelineProfiler()
        with self.assertRaisesRegex(ValueError, 'secret'):
            with p.profile_context('outer', profiler=profiler):
                with p.profile_context('inner', profiler=profiler):
                    raise ValueError('secret')
        summary = profiler.get_summary()
        self.assertEqual(summary['stage_count'], 2)
        self.assertEqual(summary['total_duration_ms'], profiler.stages[-1].duration_ms)
        self.assertEqual([s['status'] for s in summary['stages']], ['error', 'error'])
        self.assertNotIn('secret', str(summary))
        self.assertEqual(profiler._stack.get(), ())

    def test_resource_metrics(self):
        profiler = p.PipelineProfiler()
        start = {"rss_mb": 100.0, "system_cpu_time": 50.0, "accelerator": "mps", "allocated_mb": 20.0, "driver_mb": 30.0}
        end = {"rss_mb": 104.5, "system_cpu_time": 54.0, "accelerator": "mps", "allocated_mb": 23.0, "driver_mb": 35.0}
        with patch.object(p.PipelineProfiler, '_resource_snapshot', side_effect=[start, end]), patch.object(p.time, 'perf_counter', side_effect=[10., 12.]), patch.object(p.os, 'times', side_effect=[SimpleNamespace(user=1., system=1.), SimpleNamespace(user=2., system=2.)]):
            with p.profile_context('resources', profiler=profiler):
                pass
        stage = profiler.stages[0]
        self.assertEqual(stage.accelerator, 'mps')
        self.assertEqual(stage.process_rss_delta_mb, 4.5)
        self.assertEqual(stage.accelerator_allocated_delta_mb, 3.0)
        self.assertEqual(stage.accelerator_driver_delta_mb, 5.0)

    def test_disabled_and_repeated_runs(self):
        @p.profile_stage('child')
        def child():
            return 42
        @p.profile_run
        def run(text, enable_profiling=False):
            return {'value': child()}
        self.assertNotIn('profiling', run('synthetic'))
        first = run('synthetic', True)
        second = run('synthetic', True)
        self.assertEqual(first['profiling']['stage_count'], 2)
        self.assertEqual(second['profiling']['stage_count'], 2)
        self.assertEqual(first['value'], 42)

    def test_json_parent_total_and_isolation(self):
        @p.profile_run
        def run(text, enable_profiling=False):
            return {}
        @p.profile_run
        def run_json(data, enable_profiling=False):
            return {'profiling': {'per_field': [run(value, enable_profiling)['profiling'] for value in data]}}
        result = run_json(['synthetic', 'synthetic'], True)['profiling']
        self.assertEqual(len(result['per_field']), 2)
        self.assertEqual(result['aggregate']['stage_count'], 3)
        self.assertGreaterEqual(result['aggregate']['total_duration_ms'], sum(f['total_duration_ms'] for f in result['per_field']))
