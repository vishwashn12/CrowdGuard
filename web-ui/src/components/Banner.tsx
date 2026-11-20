/* Exactly one message: the most urgent actionable alert. */
import { Siren, AlertTriangle, ChevronRight } from 'lucide-react';
import type { Snapshot } from '../lib/types';

export function Banner({ snap, onSelect }: { snap: Snapshot; onSelect: (id: string) => void }) {
  if (snap.session.state !== 'running') return null;
  const actionable = (snap.alerts || []).filter((a) => a.severity === 'EMERGENCY' || a.severity === 'ALERT');
  if (!actionable.length) return null;
  const top = actionable[0];
  const gate = top.gate_id && top.gate_id !== 'ALL' ? top.gate_id : null;
  const Icon = top.severity === 'EMERGENCY' ? Siren : AlertTriangle;
  return (
    <div className="banner" data-sev={top.severity} role="alert">
      <Icon size={20} />
      <span className="banner-sev">{top.severity}</span>
      {gate && <span className="banner-gate">{gate}</span>}
      <strong>{top.title}</strong>
      <span className="banner-do">{top.action}</span>
      <span className="spacer" />
      {actionable.length > 1 && <span className="banner-more">+{actionable.length - 1} more</span>}
      {gate && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => onSelect(gate)}>
          Open gate <ChevronRight size={14} />
        </button>
      )}
    </div>
  );
}
