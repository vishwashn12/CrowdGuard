import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';
import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react';

type Kind = 'ok' | 'err' | 'warn' | 'info';
interface Toast { id: number; kind: Kind; title: string; body?: string }
type Push = (kind: Kind, title: string, body?: string) => void;

const Ctx = createContext<Push>(() => {});

export function useToast(): Push {
  return useContext(Ctx);
}

let nextId = 1;
const ICON = { ok: CheckCircle2, err: AlertTriangle, warn: AlertTriangle, info: Info };

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => setItems((xs) => xs.filter((t) => t.id !== id)), []);

  const push = useCallback<Push>((kind, title, body) => {
    const id = nextId++;
    setItems((xs) => [...xs.slice(-3), { id, kind, title, body }]);
    setTimeout(() => dismiss(id), kind === 'err' ? 9000 : 5500);
  }, [dismiss]);

  return (
    <Ctx.Provider value={push}>
      {children}
      <div className="toasts" aria-live="polite">
        {items.map((t) => {
          const I = ICON[t.kind];
          return (
            <div key={t.id} className={`toast toast-${t.kind}`}>
              <I size={18} className="toast-icon" />
              <div className="toast-text">
                <strong>{t.title}</strong>
                {t.body && <span>{t.body}</span>}
              </div>
              <button className="icon-btn" onClick={() => dismiss(t.id)} aria-label="Dismiss">
                <X size={14} />
              </button>
            </div>
          );
        })}
      </div>
    </Ctx.Provider>
  );
}
