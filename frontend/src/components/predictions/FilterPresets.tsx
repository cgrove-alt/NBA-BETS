import { useState } from 'react';
import { Save, Bookmark, Trash2, Check, X } from 'lucide-react';
import type { FilterPreset } from '../../lib/types';

interface FilterPresetsProps {
  presets: FilterPreset[];
  currentFiltersActive: boolean;
  onSavePreset: (name: string, description?: string) => void;
  onLoadPreset: (presetId: string) => void;
  onDeletePreset: (presetId: string) => void;
}

export function FilterPresets({
  presets,
  currentFiltersActive,
  onSavePreset,
  onLoadPreset,
  onDeletePreset,
}: FilterPresetsProps) {
  const [isCreating, setIsCreating] = useState(false);
  const [presetName, setPresetName] = useState('');
  const [presetDescription, setPresetDescription] = useState('');

  const handleSave = () => {
    if (presetName.trim()) {
      onSavePreset(presetName.trim(), presetDescription.trim() || undefined);
      setPresetName('');
      setPresetDescription('');
      setIsCreating(false);
    }
  };

  const handleCancel = () => {
    setPresetName('');
    setPresetDescription('');
    setIsCreating(false);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-xs text-text-secondary font-medium">Filter Presets</label>
        {!isCreating && currentFiltersActive && (
          <button
            onClick={() => setIsCreating(true)}
            className="flex items-center gap-1 px-2 py-1 text-xs text-accent-primary hover:bg-accent-primary/10 rounded transition-colors"
          >
            <Save size={12} />
            Save Current
          </button>
        )}
      </div>

      {/* Create new preset form */}
      {isCreating && (
        <div className="bg-bg-tertiary border border-border rounded-lg p-3 space-y-2">
          <input
            type="text"
            placeholder="Preset name (required)"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
            className="w-full px-2 py-1.5 text-xs bg-bg-primary border border-border rounded text-text-primary focus:outline-none focus:border-accent-primary"
            autoFocus
          />
          <input
            type="text"
            placeholder="Description (optional)"
            value={presetDescription}
            onChange={(e) => setPresetDescription(e.target.value)}
            className="w-full px-2 py-1.5 text-xs bg-bg-primary border border-border rounded text-text-primary focus:outline-none focus:border-accent-primary"
          />
          <div className="flex gap-2">
            <button
              onClick={handleSave}
              disabled={!presetName.trim()}
              className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-accent-primary text-white rounded hover:bg-accent-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Check size={12} />
              Save
            </button>
            <button
              onClick={handleCancel}
              className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-bg-primary border border-border text-text-secondary rounded hover:bg-bg-hover transition-colors"
            >
              <X size={12} />
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Presets list */}
      {presets.length > 0 ? (
        <div className="space-y-1.5">
          {presets.map((preset) => (
            <div
              key={preset.id}
              className="flex items-center gap-2 p-2 bg-bg-tertiary border border-border rounded-lg hover:border-accent-primary/30 transition-colors group"
            >
              <button
                onClick={() => onLoadPreset(preset.id)}
                className="flex-1 flex items-start gap-2 text-left"
              >
                <Bookmark size={14} className="text-accent-primary shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-text-primary truncate">
                    {preset.name}
                  </div>
                  {preset.description && (
                    <div className="text-xs text-text-muted truncate mt-0.5">
                      {preset.description}
                    </div>
                  )}
                </div>
              </button>
              <button
                onClick={() => onDeletePreset(preset.id)}
                className="opacity-0 group-hover:opacity-100 p-1 text-accent-danger hover:bg-accent-danger/10 rounded transition-all"
                aria-label="Delete preset"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-xs text-text-muted text-center py-4">
          No saved presets. Apply filters and save to create a preset.
        </div>
      )}
    </div>
  );
}
