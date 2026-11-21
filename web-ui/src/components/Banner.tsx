/* At most one line across the top: the most urgent critical instruction,
 * worded exactly as staff should hear it. */
import { ChevronRight, Radio } from 'lucide-react';
import type { Snapshot } from '../lib/types';
import { InstructionText, VerbIcon } from './bits';

export function Banner({ snap, onSelect }: { snap: Snapshot; onSelect: (id: string) => void }) {
  if (snap.session.state !== 'running') return null;
  const critical = [...(snap.commands ?? []), ...(snap.incident_list ?? [])].filter((i) => i.level === 'critical');
  if (!critical.length) return null;
  const top = critical[0];
  return (
    <div className="banner" role="alert">
      <span className="banner-icon"><VerbIcon verb={top.verb} kind={top.kind} size={20} /></span>
      <strong className="banner-head"><InstructionText item={top} /></strong>
      <span className="banner-reason">{top.reason}</span>
      <span className="banner-radio"><Radio size={14} /> {top.radio}</span>
      <span className="spacer" />
      {critical.length > 1 && <span className="banner-more">+{critical.length - 1} more</span>}
      <button type="button" className="btn btn-ghost btn-sm" onClick={() => onSelect(top.gate_id)}>
        Show {top.gate_id} <ChevronRight size={14} />
      </button>
    </div>
  );
}
