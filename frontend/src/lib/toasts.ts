import { create } from 'zustand';

interface ToastItem { id: number; msg: string; }

export const useToasts = create<{ list: ToastItem[]; push: (m: string) => void; drop: (id: number) => void }>((set, get) => ({
  list: [],
  push: (msg) => {
    const id = Date.now() + Math.random();
    set({ list: [...get().list, { id, msg }] });
    setTimeout(() => get().drop(id), 2800);
  },
  drop: (id) => set({ list: get().list.filter(t => t.id !== id) }),
}));

export function toast(msg: string) { useToasts.getState().push(msg); }