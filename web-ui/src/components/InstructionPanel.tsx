/* The operator's to-do list, in two channels:
 *   COMMANDS  - what to do with the crowd at a gate (one per gate)
 *   INCIDENTS - where to send staff
 * Each row reads as "VERB GATE → TARGET · reason". Tapping a row opens the
 * one-line detail and the radio script to read out to gate staff. */
import { useState } from 'react';
import { ChevronDown, Radio, ShieldCheck } from 'lucide-react';
import type { Instruction, LogEntry, Snapshot } from '../lib/types';
import { fmtAge } from '../lib/format';
import { InstructionText, VerbIcon } from './bits';

function Row({ item, now, onSelect, open, onToggle }: {
  item: Instruction; now: number; onSelect: (id: string) => void; open: boolean; onToggle: () => void;
}) {
  return (
    <li className={`ins ${open ? 'is-open' : ''}`} data-level={item.level} data-verb={item.verb}>
      <button type="button" className="ins-main" onClick={onToggle} aria-expanded={open}>
        <span className="ins-icon"><VerbIcon verb={item.verb} kind={item.kind} size={18} /></span>
        <span className="ins-text">
          <span className="ins-head"><InstructionText item={item} /></span>
          <span className="ins-reason">{item.reason}</span>
        </span>
        <span className="ins-age">{fmtAge(now - item.since)}</span>
        <ChevronDown size={15} className="ins-chev" />
      </button>
      {open && (
        <div className="ins-more">
          <p className="ins-radio"><Radio size={14} /> “{item.radio}”</p>
          <p className="ins-detail">{item.detail}</p>
          <button type="button" className="btn btn-sm btn-ghost" onClick={() => onSelect(item.gate_id)}>
            Show {item.gate_id}
          </button>
        </div>
      )}
    </li>
  );
}

function Section({ title, items, empty, now, onSelect }: {
  title: string; items: Instruction[]; empty: string; now: number; onSelect: (id: string) => void;
}) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <section className="ins-section">
      <h3>{title}{items.length > 0 && <span className="count">{items.length}</span>}</h3>
      {items.length === 0 ? <p className="ins-empty">{empty}</p> : (
        <ul className="ins-list">
          {items.map((it) => {
            const key = `${it.channel}-${it.gate_id}-${it.verb}-${it.kind ?? ''}`;
            return <Row key={key} item={it} now={now} onSelect={onSelect}
              open={open === key} onToggle={() => setOpen(open === key ? null : key)} />;
          })}
        </ul>
      )}
    </section>
  );
}

function LogList({ log, now }: { log: LogEntry[]; now: number }) {
  if (!log.length) return <p className="ins-empty pad">Nothing logged in this session.</p>;
  return (
    <ul className="log-list">
      {log.map((e, i) => (
        <li key={`${e.gate_id}-${e.verb}-${e.time}-${i}`} data-level={e.level}>
          <span className="log-icon"><VerbIcon verb={e.verb} kind={e.kind} size={14} /></span>
          <span className="log-head">
            <span className="verb">{e.verb_label}</span> {e.gate_id}{e.target && <> → {e.target}</>}
          </span>
          <span className="log-reason">{e.reason}</span>
          <span className="log-age">{fmtAge(now - e.time)} ago</span>
        </li>
      ))}
    </ul>
  );
}

export function InstructionPanel({ snap, onSelect }: { snap: Snapshot; onSelect: (id: string) => void }) {
  const [tab, setTab] = useState<'now' | 'log'>('now');
  const running = snap.session.state === 'running';
  const commands = snap.commands ?? [];
  const incidents = snap.incident_list ?? [];
  const now = snap.timestamp;
  const quiet = running && !commands.length && !incidents.length;

  return (
    <section className="panel instructions">
      <header className="panel-head">
        <h2>Instructions</h2>
        <span className="spacer" />
        <div className="seg seg-sm" role="tablist">
          <button type="button" className={tab === 'now' ? 'on' : ''} onClick={() => setTab('now')}>Now</button>
          <button type="button" className={tab === 'log' ? 'on' : ''} onClick={() => setTab('log')}>Log</button>
        </div>
      </header>

      <div className="panel-scroll">
        {tab === 'log' && <LogList log={snap.instruction_log ?? []} now={now} />}
        {tab === 'now' && !running && (
          <div className="all-clear">
            <ShieldCheck size={34} strokeWidth={1.5} />
            <strong>Not monitoring</strong>
            <span>Instructions appear here once monitoring starts.</span>
          </div>
        )}
        {tab === 'now' && quiet && (
          <div className="all-clear">
            <ShieldCheck size={34} strokeWidth={1.5} />
            <strong>All clear</strong>
            <span>No instructions. Keep watching.</span>
          </div>
        )}
        {tab === 'now' && running && !quiet && (
          <>
            <Section title="Commands" items={commands} empty="No crowd commands." now={now} onSelect={onSelect} />
            <Section title="Incidents" items={incidents} empty="No incidents." now={now} onSelect={onSelect} />
          </>
        )}
      </div>
    </section>
  );
}
