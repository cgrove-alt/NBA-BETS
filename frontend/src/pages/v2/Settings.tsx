import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Settings as SettingsIcon,
  DollarSign,
  Target,
  Bell,
  Shield,
  Palette,
  User,
  ChevronRight,
  Save,
  Zap,
  Check,
  AlertCircle,
} from 'lucide-react';
import { ResponsiveLayout } from '../../components/v2/ResponsiveLayout';
import { Card } from '../../components/v2/Card';
import { Button } from '../../components/v2/Button';
import { Badge } from '../../components/v2/Badge';
import { fetchSettings, updateSettings as updateSettingsApi } from '../../lib/api';
import { useBankroll } from '../../hooks/useBankroll';

interface StrategySettings {
  bankroll: number;
  defaultBetSize: number;
  betSizeType: 'fixed' | 'percentage' | 'kelly';
  minConfidence: number;
  minEdge: number;
  maxBetsPerDay: number;
  notifications: {
    topPicks: boolean;
    resultUpdates: boolean;
    bankrollAlerts: boolean;
  };
}

/**
 * Settings - Configure betting strategy and preferences
 */
export function Settings() {
  const queryClient = useQueryClient();
  const { bankrollData } = useBankroll();

  // Load settings from backend
  const { data: serverSettings } = useQuery({
    queryKey: ['settings'],
    queryFn: fetchSettings,
    staleTime: 60 * 1000,
  });

  // Derive default settings, merging server data when available.
  // useMemo ensures we don't create a new object every render.
  const defaultSettings = useMemo<StrategySettings>(() => {
    const base: StrategySettings = {
      bankroll: 5000,
      defaultBetSize: 100,
      betSizeType: 'fixed',
      minConfidence: 55,
      minEdge: 5,
      maxBetsPerDay: 10,
      notifications: {
        topPicks: true,
        resultUpdates: true,
        bankrollAlerts: true,
      },
    };
    if (serverSettings) {
      return {
        ...base,
        bankroll: serverSettings.bankroll,
        defaultBetSize: serverSettings.default_bet_size,
        betSizeType: serverSettings.bet_size_type as 'fixed' | 'percentage' | 'kelly',
        minConfidence: serverSettings.min_confidence,
        minEdge: serverSettings.min_edge,
        maxBetsPerDay: serverSettings.max_bets_per_day,
      };
    }
    return base;
  }, [serverSettings]);

  // Local edits on top of server-derived defaults.
  // Tracks only fields the user has changed since the last save.
  const [localOverrides, setLocalOverrides] = useState<Partial<StrategySettings>>({});

  // Effective settings = server defaults merged with any local edits
  const settings: StrategySettings = useMemo(
    () => ({ ...defaultSettings, ...localOverrides }),
    [defaultSettings, localOverrides],
  );

  const [hasChanges, setHasChanges] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');

  // Save mutation
  const saveMutation = useMutation({
    mutationFn: () =>
      updateSettingsApi({
        bankroll: settings.bankroll,
        default_bet_size: settings.defaultBetSize,
        bet_size_type: settings.betSizeType,
        min_confidence: settings.minConfidence,
        min_edge: settings.minEdge,
        max_bets_per_day: settings.maxBetsPerDay,
      }),
    onSuccess: () => {
      setLocalOverrides({});
      setHasChanges(false);
      setSaveStatus('success');
      queryClient.invalidateQueries({ queryKey: ['settings'] });
      queryClient.invalidateQueries({ queryKey: ['bankroll'] });
      setTimeout(() => setSaveStatus('idle'), 2000);
    },
    onError: () => {
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 3000);
    },
  });

  const updateSettingField = <K extends keyof StrategySettings>(
    key: K,
    value: StrategySettings[K]
  ) => {
    setLocalOverrides((prev) => ({ ...prev, [key]: value }));
    setHasChanges(true);
    setSaveStatus('idle');
  };

  const updateNotification = (key: keyof StrategySettings['notifications'], value: boolean) => {
    setLocalOverrides((prev) => {
      const currentNotifications = {
        ...settings.notifications,
        ...(prev.notifications || {}),
      };
      return {
        ...prev,
        notifications: { ...currentNotifications, [key]: value },
      };
    });
    setHasChanges(true);
  };

  const handleSave = () => {
    saveMutation.mutate();
  };

  return (
    <ResponsiveLayout bankroll={bankrollData} activePage="settings">
      <div className="space-y-6 pb-20 md:pb-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">Settings</h1>
            <p className="text-sm text-text-muted mt-1">
              Configure your betting strategy
            </p>
          </div>
          {(hasChanges || saveStatus !== 'idle') && (
            <Button
              variant="action"
              size="sm"
              icon={
                saveStatus === 'success' ? <Check className="w-4 h-4" /> :
                saveStatus === 'error' ? <AlertCircle className="w-4 h-4" /> :
                <Save className="w-4 h-4" />
              }
              onClick={handleSave}
              disabled={saveMutation.isPending || saveStatus === 'success'}
            >
              {saveMutation.isPending ? 'Saving...' :
               saveStatus === 'success' ? 'Saved' :
               saveStatus === 'error' ? 'Error - Retry' :
               'Save Changes'}
            </Button>
          )}
        </div>

        {/* Bankroll Management */}
        <SettingsSection
          title="Bankroll Management"
          icon={<DollarSign className="w-5 h-5" />}
        >
          <div className="space-y-4">
            {/* Bankroll Amount */}
            <div>
              <label className="block text-sm text-text-primary mb-2">
                Total Bankroll
              </label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted">
                  $
                </span>
                <input
                  type="number"
                  value={settings.bankroll}
                  onChange={(e) => updateSettingField('bankroll', Number(e.target.value))}
                  className="w-full bg-bg-tertiary border border-border rounded-lg px-8 py-3 text-text-primary focus:outline-none focus:border-[#00d4ff] focus:ring-1 focus:ring-[#00d4ff]"
                />
              </div>
            </div>

            {/* Bet Sizing */}
            <div>
              <label className="block text-sm text-text-primary mb-2">
                Bet Sizing Strategy
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { value: 'fixed', label: 'Fixed', desc: 'Same amount every bet' },
                  { value: 'percentage', label: 'Percentage', desc: '% of bankroll' },
                  { value: 'kelly', label: 'Kelly', desc: 'Optimal sizing' },
                ].map((option) => (
                  <button
                    key={option.value}
                    onClick={() => updateSettingField('betSizeType', option.value as StrategySettings['betSizeType'])}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      settings.betSizeType === option.value
                        ? 'border-[#00d4ff] bg-[rgba(0,212,255,0.1)]'
                        : 'border-border bg-bg-tertiary hover:border-[rgba(255,255,255,0.1)]'
                    }`}
                  >
                    <div className="text-sm font-medium text-text-primary">
                      {option.label}
                    </div>
                    <div className="text-xs text-text-muted mt-1">{option.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Default Bet Size */}
            <div>
              <label className="block text-sm text-text-primary mb-2">
                Default Bet Size
                {settings.betSizeType === 'percentage' && ' (%)'}
              </label>
              <div className="relative">
                {settings.betSizeType !== 'percentage' && (
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted">
                    $
                  </span>
                )}
                <input
                  type="number"
                  value={settings.defaultBetSize}
                  onChange={(e) => updateSettingField('defaultBetSize', Number(e.target.value))}
                  className={`w-full bg-bg-tertiary border border-border rounded-lg py-3 text-text-primary focus:outline-none focus:border-[#00d4ff] focus:ring-1 focus:ring-[#00d4ff] ${
                    settings.betSizeType === 'percentage' ? 'px-4' : 'px-8'
                  }`}
                />
                {settings.betSizeType === 'percentage' && (
                  <span className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted">
                    %
                  </span>
                )}
              </div>
            </div>

            {/* Max Bets Per Day */}
            <div>
              <label className="block text-sm text-text-primary mb-2">
                Max Bets Per Day
              </label>
              <input
                type="number"
                value={settings.maxBetsPerDay}
                onChange={(e) => updateSettingField('maxBetsPerDay', Number(e.target.value))}
                className="w-full bg-bg-tertiary border border-border rounded-lg px-4 py-3 text-text-primary focus:outline-none focus:border-[#00d4ff] focus:ring-1 focus:ring-[#00d4ff]"
              />
            </div>
          </div>
        </SettingsSection>

        {/* Prediction Filters */}
        <SettingsSection
          title="Prediction Filters"
          icon={<Target className="w-5 h-5" />}
        >
          <div className="space-y-4">
            {/* Min Confidence */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-text-primary">
                  Minimum Confidence
                </label>
                <Badge variant="default">{settings.minConfidence}%</Badge>
              </div>
              <input
                type="range"
                min="40"
                max="80"
                step="5"
                value={settings.minConfidence}
                onChange={(e) => updateSettingField('minConfidence', Number(e.target.value))}
                className="w-full accent-[#00d4ff]"
              />
              <div className="flex justify-between text-xs text-text-muted mt-1">
                <span>40%</span>
                <span>80%</span>
              </div>
            </div>

            {/* Min Edge */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-text-primary">Minimum Edge</label>
                <Badge variant="default">{settings.minEdge}%</Badge>
              </div>
              <input
                type="range"
                min="0"
                max="20"
                step="1"
                value={settings.minEdge}
                onChange={(e) => updateSettingField('minEdge', Number(e.target.value))}
                className="w-full accent-[#00d4ff]"
              />
              <div className="flex justify-between text-xs text-text-muted mt-1">
                <span>0%</span>
                <span>20%</span>
              </div>
            </div>
          </div>
        </SettingsSection>

        {/* Notifications */}
        <SettingsSection
          title="Notifications"
          icon={<Bell className="w-5 h-5" />}
        >
          <div className="space-y-3">
            <ToggleSetting
              label="Top Picks Alerts"
              description="Get notified when new top picks are available"
              enabled={settings.notifications.topPicks}
              onChange={(v) => updateNotification('topPicks', v)}
            />
            <ToggleSetting
              label="Result Updates"
              description="Get notified when your bets are graded"
              enabled={settings.notifications.resultUpdates}
              onChange={(v) => updateNotification('resultUpdates', v)}
            />
            <ToggleSetting
              label="Bankroll Alerts"
              description="Get notified about significant bankroll changes"
              enabled={settings.notifications.bankrollAlerts}
              onChange={(v) => updateNotification('bankrollAlerts', v)}
            />
          </div>
        </SettingsSection>

        {/* Quick Links */}
        <SettingsSection
          title="More Options"
          icon={<SettingsIcon className="w-5 h-5" />}
        >
          <div className="space-y-2">
            <LinkButton icon={<Palette />} label="Appearance" />
            <LinkButton icon={<User />} label="Account" />
            <LinkButton icon={<Shield />} label="Privacy" />
          </div>
        </SettingsSection>

        {/* App Info */}
        <Card variant="glass" className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[rgba(0,255,136,0.1)]">
              <Zap className="w-5 h-5 text-[#00ff88]" />
            </div>
            <div>
              <div className="font-semibold text-text-primary">The Oracle</div>
              <div className="text-xs text-text-muted">Version 2.0.0 • NBA Betting Terminal</div>
            </div>
          </div>
        </Card>
      </div>
    </ResponsiveLayout>
  );
}

/**
 * Settings Section - Groups related settings
 */
function SettingsSection({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <div className="p-4 border-b border-border flex items-center gap-3">
        <div className="text-[#00d4ff]">{icon}</div>
        <h2 className="font-semibold text-text-primary">{title}</h2>
      </div>
      <div className="p-4">{children}</div>
    </Card>
  );
}

/**
 * Toggle Setting - On/off switch with description
 */
function ToggleSetting({
  label,
  description,
  enabled,
  onChange,
}: {
  label: string;
  description: string;
  enabled: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between p-3 rounded-lg bg-bg-tertiary">
      <div>
        <div className="text-sm font-medium text-text-primary">{label}</div>
        <div className="text-xs text-text-muted mt-0.5">{description}</div>
      </div>
      <button
        onClick={() => onChange(!enabled)}
        className={`relative w-12 h-6 rounded-full transition-colors ${
          enabled ? 'bg-[#00ff88]' : 'bg-bg-secondary'
        }`}
      >
        <span
          className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
            enabled ? 'left-7' : 'left-1'
          }`}
        />
      </button>
    </div>
  );
}

/**
 * Link Button - Navigation item
 */
function LinkButton({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <button className="w-full flex items-center justify-between p-3 rounded-lg bg-bg-tertiary hover:bg-bg-card-hover transition-colors">
      <div className="flex items-center gap-3">
        <span className="text-text-muted">{icon}</span>
        <span className="text-sm text-text-primary">{label}</span>
      </div>
      <ChevronRight className="w-4 h-4 text-text-muted" />
    </button>
  );
}
