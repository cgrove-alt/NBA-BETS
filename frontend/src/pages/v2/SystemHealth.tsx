import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  Server,
  Database,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  Cpu,
} from 'lucide-react';
import { ResponsiveLayout } from '../../components/v2/ResponsiveLayout';
import { Card } from '../../components/v2/Card';
import { Badge } from '../../components/v2/Badge';
import { fetchSystemHealth } from '../../lib/api';
import { useBankroll } from '../../hooks/useBankroll';
import type { AgentStatusData, ModelStatusData } from '../../lib/types';

const AGENT_DISPLAY_NAMES: Record<string, string> = {
  pregame: 'Pre-Game Intel',
  postgame: 'Post-Game Analysis',
  odds_monitor: 'Odds Monitor',
  orchestrator: 'Prediction Orchestrator',
  watchdog: 'Model Watchdog',
  briefing: 'Daily Briefing',
};

/**
 * SystemHealth - Monitor agent status, model freshness, and data freshness
 */
export function SystemHealth() {
  const { bankrollData } = useBankroll();

  const { data, isLoading, error } = useQuery({
    queryKey: ['systemHealth'],
    queryFn: fetchSystemHealth,
    staleTime: 30 * 1000,
    refetchInterval: 60 * 1000,
  });

  const overallStatus = data?.overall_status || 'unknown';

  return (
    <ResponsiveLayout bankroll={bankrollData} activePage="health">
      <div className="space-y-6 pb-20 md:pb-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">System Health</h1>
            <p className="text-sm text-text-muted mt-1">Monitor agents, models, and data freshness</p>
          </div>
          <OverallStatusBadge status={overallStatus} />
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-text-muted" />
          </div>
        ) : error ? (
          <Card className="p-8 text-center">
            <AlertTriangle className="w-8 h-8 text-[#ff8800] mx-auto mb-2" />
            <p className="text-text-muted">Failed to load system health data</p>
          </Card>
        ) : (
          <>
            {/* Agent Status Cards */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <Cpu className="w-5 h-5 text-[#00d4ff]" />
                <h2 className="text-lg font-bold text-text-primary">Agents</h2>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(data?.agents || {}).map(([name, status]) => (
                  <AgentCard key={name} name={name} status={status} />
                ))}
              </div>
            </section>

            {/* Model Freshness */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <Server className="w-5 h-5 text-[#00d4ff]" />
                <h2 className="text-lg font-bold text-text-primary">Models</h2>
              </div>
              <Card>
                <div className="divide-y divide-border">
                  {(data?.models || []).length > 0 ? (
                    data!.models.map((model) => (
                      <ModelRow key={model.filename} model={model} />
                    ))
                  ) : (
                    <div className="p-4 text-center text-text-muted">No model files found</div>
                  )}
                </div>
              </Card>
            </section>

            {/* Data Freshness */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <Database className="w-5 h-5 text-[#00d4ff]" />
                <h2 className="text-lg font-bold text-text-primary">Data Freshness</h2>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {Object.entries(data?.data_freshness || {}).map(([key, value]) => (
                  <FreshnessCard key={key} label={formatFreshnessLabel(key)} timestamp={value} />
                ))}
              </div>
            </section>
          </>
        )}
      </div>
    </ResponsiveLayout>
  );
}

function OverallStatusBadge({ status }: { status: string }) {
  if (status === 'healthy') {
    return (
      <Badge variant="success" glow>
        <CheckCircle className="w-3 h-3 mr-1" /> Healthy
      </Badge>
    );
  }
  if (status === 'degraded') {
    return (
      <Badge variant="warning">
        <AlertTriangle className="w-3 h-3 mr-1" /> Degraded
      </Badge>
    );
  }
  if (status === 'critical') {
    return (
      <Badge variant="danger" glow>
        <XCircle className="w-3 h-3 mr-1" /> Critical
      </Badge>
    );
  }
  return <Badge variant="default">Unknown</Badge>;
}

function AgentCard({ name, status }: { name: string; status: AgentStatusData }) {
  const displayName = AGENT_DISPLAY_NAMES[name] || name;
  const isHealthy = status.consecutive_failures === 0 && status.last_status === 'completed';
  const isCritical = status.consecutive_failures >= 3;
  const isDegraded = status.consecutive_failures >= 1;

  const statusColor = isCritical
    ? 'border-[#ff3355]'
    : isDegraded
    ? 'border-[#ff8800]'
    : isHealthy
    ? 'border-[#00ff88]'
    : 'border-border';

  const StatusIcon = isCritical
    ? XCircle
    : isDegraded
    ? AlertTriangle
    : isHealthy
    ? CheckCircle
    : Clock;

  const statusIconColor = isCritical
    ? 'text-[#ff3355]'
    : isDegraded
    ? 'text-[#ff8800]'
    : isHealthy
    ? 'text-[#00ff88]'
    : 'text-text-muted';

  return (
    <Card className={`border-l-2 ${statusColor}`}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-text-primary text-sm">{displayName}</h3>
          <StatusIcon className={`w-4 h-4 ${statusIconColor}`} />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-text-muted">Last Run</span>
            <span className="text-text-secondary">
              {status.last_run ? formatTimeAgo(status.last_run) : 'Never'}
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-text-muted">Status</span>
            <span className="text-text-secondary">{status.last_status || 'N/A'}</span>
          </div>
          {status.consecutive_failures > 0 && (
            <div className="flex justify-between text-xs">
              <span className="text-text-muted">Failures</span>
              <span className="text-[#ff3355]">{status.consecutive_failures} consecutive</span>
            </div>
          )}
          <div className="flex justify-between text-xs">
            <span className="text-text-muted">Tokens Today</span>
            <span className="text-text-secondary">{status.tokens_used_today.toLocaleString()}</span>
          </div>
        </div>
      </div>
    </Card>
  );
}

function ModelRow({ model }: { model: ModelStatusData }) {
  const isStale = model.age_days > 30;
  const isWarning = model.age_days > 14;

  return (
    <div className="p-3 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <Activity className={`w-4 h-4 ${isStale ? 'text-[#ff3355]' : isWarning ? 'text-[#ff8800]' : 'text-[#00ff88]'}`} />
        <span className="text-sm text-text-primary font-mono">{model.filename}</span>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-xs text-text-muted">{model.age_days}d old</span>
        <Badge
          variant={isStale ? 'danger' : isWarning ? 'warning' : 'success'}
          size="sm"
        >
          {isStale ? 'Stale' : isWarning ? 'Aging' : 'Fresh'}
        </Badge>
      </div>
    </div>
  );
}

function FreshnessCard({ label, timestamp }: { label: string; timestamp: string | null }) {
  return (
    <Card>
      <div className="p-4">
        <div className="flex items-center gap-2 text-text-muted mb-2">
          <Clock className="w-4 h-4" />
          <span className="text-xs uppercase tracking-wider">{label}</span>
        </div>
        <div className="text-sm font-semibold text-text-primary">
          {timestamp ? formatTimeAgo(timestamp) : 'No data'}
        </div>
      </div>
    </Card>
  );
}

function formatTimeAgo(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  const diffHrs = Math.floor(diffMin / 60);
  const diffDays = Math.floor(diffHrs / 24);

  if (diffMin < 1) return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHrs < 24) return `${diffHrs}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatFreshnessLabel(key: string): string {
  const labels: Record<string, string> = {
    last_predictions: 'Last Predictions',
    last_odds_fetch: 'Last Odds Fetch',
    last_bdl_call: 'Last Stats API',
  };
  return labels[key] || key.replace(/_/g, ' ');
}
