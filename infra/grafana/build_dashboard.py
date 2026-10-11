"""Writes the Grafana dashboard (infra/grafana/dashboards/llm-gateway.json); edit this, not the JSON.

Usage: uv run python infra/grafana/build_dashboard.py infra/grafana/dashboards/llm-gateway.json
Grafana reads the file at start: docker compose restart grafana."""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
DS = {"type": "prometheus", "uid": "prometheus"}
# Reference categorical palette, dark-surface steps (Grafana's default theme is dark), fixed per entity.
PROVIDER_COLORS = {"openai": "#3987e5", "anthropic": "#d95926", "groq": "#199e70"}
STATE_COLORS = {"stream": "#3987e5", "delayed": "#c98500", "dead": "#e66767"}

panels, next_id = [], iter(range(1, 100))


def color_overrides(colors):
    return [{"matcher": {"id": "byName", "options": name},
             "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]}
            for name, color in colors.items()]


def timeseries(title, description, targets, x, y, w=12, h=8, unit="short", overrides=(), minimum=0, maximum=None,
               draw="line", no_value=None):
    defaults = {"unit": unit, "min": minimum, "color": {"mode": "palette-classic"},
                "custom": {"drawStyle": draw, "lineWidth": 2, "fillOpacity": 0, "pointSize": 8,
                           "showPoints": "never", "spanNulls": False, "axisBorderShow": False,
                           "gradientMode": "none", "lineInterpolation": "linear"}}
    if no_value is not None:
        defaults["noValue"] = no_value
    if maximum is not None:
        defaults["max"] = maximum
    panels.append({
        "id": next(next_id), "type": "timeseries", "title": title, "description": description, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": defaults, "overrides": list(overrides)},
        "options": {"legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
        "targets": [{"refId": chr(65 + i), "datasource": DS, "expr": expr, "legendFormat": legend, "range": True}
                    for i, (expr, legend) in enumerate(targets)],
    })


RATE = "$__rate_interval"

timeseries(
    "Provider calls per second",
    "Calls the gateway made to each provider, every outcome included. Attempts refused by an open circuit "
    "are not calls and are left out; a failover counts one call per provider tried.",
    [(f'sum by (provider) (rate(llm_gateway_provider_attempts_total{{outcome!="circuit_open"}}[{RATE}]))',
      "{{provider}}")],
    0, 0, unit="reqps", overrides=color_overrides(PROVIDER_COLORS))

timeseries(
    "Client error rate",
    "Share of chat requests that ended in an upstream error after routing, failover and the breaker. "
    "Requests rejected for a bad header or body are excluded.",
    [(f'(sum(rate(llm_gateway_requests_total{{outcome="upstream_error"}}[{RATE}])) or vector(0)) / '
      f'sum(rate(llm_gateway_requests_total{{outcome=~"ok|upstream_error"}}[{RATE}]))', "client errors")],
    12, 0, unit="percentunit", maximum=1)

timeseries(
    "Provider failure rate",
    "Share of calls to each provider that failed (any error kind), over 30 s. Read it against the client "
    "error rate: a provider can fail every call while clients see no errors, because failover answered them. "
    "While a circuit is open the provider gets only probes, so its line is sparse.",
    [('(sum by (provider) (rate(llm_gateway_provider_attempts_total{outcome!~"ok|circuit_open"}[30s]))'
      ' or 0 * sum by (provider) (rate(llm_gateway_provider_attempts_total{outcome!="circuit_open"}[30s]))) / '
      'sum by (provider) (rate(llm_gateway_provider_attempts_total{outcome!="circuit_open"}[30s]))',
      "{{provider}}")],
    0, 8, unit="percentunit", maximum=1, overrides=color_overrides(PROVIDER_COLORS))

timeseries(
    "p95 latency of answered calls",
    "From the gateway's latency histogram (buckets 0.25 s to 64 s, doubling), so the value is interpolated "
    "inside a bucket. Failed calls are not in it.",
    [(f'histogram_quantile(0.95, sum by (le, provider) (rate(llm_gateway_provider_latency_seconds_bucket[{RATE}])))',
      "{{provider}}")],
    12, 8, unit="s", overrides=color_overrides(PROVIDER_COLORS))

panels.append({
    "id": next(next_id), "type": "state-timeline", "title": "Circuit state",
    "description": "Breaker state per provider, shared by every gateway instance through Redis.",
    "datasource": DS, "gridPos": {"x": 0, "y": 16, "w": 24, "h": 5},
    "fieldConfig": {
        "defaults": {
            "color": {"mode": "fixed", "fixedColor": "#1a7f37"},
            "thresholds": {"mode": "absolute", "steps": [{"color": "#1a7f37", "value": None},
                                                         {"color": "#c98500", "value": 0.5},
                                                         {"color": "#d1242f", "value": 1.5}]},
            "mappings": [{"type": "range", "options": {"from": lo, "to": hi, "result": {
                "text": text, "color": color, "index": i}}}
                for i, (lo, hi, text, color) in enumerate([(-0.5, 0.5, "closed", "#1a7f37"),
                                                          (0.5, 1.5, "half-open", "#c98500"),
                                                          (1.5, 2.5, "open", "#d1242f")])],
            "custom": {"fillOpacity": 85, "lineWidth": 0}},
        "overrides": []},
    "options": {"showValue": "auto", "mergeValues": True, "rowHeight": 0.8, "alignValue": "left",
                "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
                "tooltip": {"mode": "single"}},
    "targets": [{"refId": "A", "datasource": DS, "expr": "max by (provider) (llm_gateway_breaker_state)",
                 "legendFormat": "{{provider}}", "range": True}],
})

timeseries(
    "Failovers per second",
    "Requests answered by a fallback model, by the model that should have answered and the one that did.",
    [(f'sum by (from_model, to_model) (rate(llm_gateway_failovers_total[{RATE}]))', "{{from_model}} -> {{to_model}}")],
    0, 21, w=8, unit="reqps", no_value="0")

timeseries(
    "Attempts refused by an open circuit",
    "Calls the gateway did not make because the provider's circuit was open: the load the breaker took off "
    "a failing provider.",
    [(f'sum by (provider) (rate(llm_gateway_provider_attempts_total{{outcome="circuit_open"}}[{RATE}]))',
      "{{provider}}")],
    8, 21, w=8, unit="reqps", overrides=color_overrides(PROVIDER_COLORS), no_value="0")

timeseries(
    "Deferred jobs",
    "Jobs in the stream (new or held by a worker), waiting for a retry, and dead (failed for good). "
    "Every instance reports the same Redis values, so the panel takes one.",
    [("max by (state) (llm_gateway_jobs)", "{{state}}")],
    16, 21, w=8, overrides=color_overrides(STATE_COLORS))

timeseries(
    "Cost per hour by tenant",
    "Spend at registry list prices, as an hourly rate over the last minute. Verification spend is not included.",
    [("sum by (tenant) (rate(llm_gateway_cost_usd_total[1m])) * 3600", "{{tenant}}")],
    0, 29, w=12, unit="currencyUSD",
    overrides=color_overrides({"tenant-a": "#9085e9", "tenant-b": "#d55181", "tenant-c": "#008300"}))

# Eight features as lines would be unreadable; one bar each, sorted, over the dashboard's time range.
panels.append({
    "id": next(next_id), "type": "bargauge", "title": "Cost per hour by feature, over the time range",
    "description": "Average hourly spend per feature across the selected time range, at registry list prices.",
    "datasource": DS, "gridPos": {"x": 12, "y": 29, "w": 12, "h": 8},
    "fieldConfig": {"defaults": {"unit": "currencyUSD", "min": 0, "decimals": 3,
                                 "color": {"mode": "fixed", "fixedColor": "#3987e5"}}, "overrides": []},
    "options": {"orientation": "horizontal", "displayMode": "basic", "showUnfilled": True, "valueMode": "text",
                "namePlacement": "top", "sizing": "auto", "minVizHeight": 16, "maxVizHeight": 36,
                "text": {"titleSize": 14, "valueSize": 14},
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
    "targets": [{"refId": "A", "datasource": DS, "instant": True, "range": False, "format": "time_series",
                 "expr": "sort_desc(sum by (feature) (rate(llm_gateway_cost_usd_total[$__range])) * 3600)",
                 "legendFormat": "{{feature}}"}],
})

dashboard = {
    "uid": "llm-gateway", "title": "LLM gateway", "tags": ["llm-gateway"], "editable": False,
    "schemaVersion": 41, "version": 1, "time": {"from": "now-15m", "to": "now"}, "refresh": "5s",
    "timepicker": {"refresh_intervals": ["2s", "5s", "10s", "30s", "1m"]}, "timezone": "browser",
    "graphTooltip": 1, "panels": panels, "templating": {"list": []}, "annotations": {"list": []},
}
OUT.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
print(f"{len(panels)} panels -> {OUT}")
