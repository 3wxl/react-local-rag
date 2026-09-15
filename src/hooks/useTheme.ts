import { useCallback, useEffect, useState } from "react";

export type ThemeMode = "light" | "dark" | "sepia";

const STORAGE_KEY = "react-local-rag:theme";

const ORDER: ThemeMode[] = ["light", "dark", "sepia"];

/** 主题元数据：标签 / 图标 key / CSS class */
export const THEME_META: Record<
  ThemeMode,
  { label: string; icon: "sun" | "moon" | "eye"; class: string }
> = {
  light: { label: "白天", icon: "sun", class: "theme-light" },
  dark: { label: "夜晚", icon: "moon", class: "theme-dark" },
  sepia: { label: "护眼", icon: "eye", class: "theme-sepia" },
};

/** 主题状态管理：localStorage 持久化 + 根元素 class 同步 */
export function useTheme() {
  const [theme, setTheme] = useState<ThemeMode>("light");

  /* 初始化：读取本地偏好，缺省跟随系统 prefers-color-scheme */
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as ThemeMode | null;
    if (stored && ORDER.includes(stored)) {
      setTheme(stored);
      return;
    }
    if (window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
      setTheme("dark");
    }
  }, []);

  /* 同步：把主题 class 写到 <html> 根元素，全 App 自动生效 */
  useEffect(() => {
    const root = document.documentElement;
    ORDER.forEach((t) => root.classList.remove(THEME_META[t].class));
    root.classList.add(THEME_META[theme].class);
    localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  /** 循环切换到下一个主题（供一键切换按钮） */
  const cycle = useCallback(() => {
    setTheme((prev) => {
      const idx = ORDER.indexOf(prev);
      return ORDER[(idx + 1) % ORDER.length];
    });
  }, []);

  return { theme, setTheme, cycle };
}
