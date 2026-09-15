import type { ThemeMode } from "../hooks/useTheme";
import { THEME_META } from "../hooks/useTheme";
import { EyeIcon, MoonIcon, SunIcon } from "./icons";

interface ThemeSwitcherProps {
  theme: ThemeMode;
  onChange: (theme: ThemeMode) => void;
}

const ICONS = {
  sun: SunIcon,
  moon: MoonIcon,
  eye: EyeIcon,
} as const;

/** 主题切换器：白天 / 夜晚 / 护眼 三选一 */
export function ThemeSwitcher({ theme, onChange }: ThemeSwitcherProps) {
  const modes = Object.keys(THEME_META) as ThemeMode[];

  return (
    <div className="flex items-center gap-1 p-1 rounded-lg bg-bg-hover">
      {modes.map((mode) => {
        const meta = THEME_META[mode];
        const Icon = ICONS[meta.icon];
        const active = theme === mode;
        return (
          <button
            key={mode}
            onClick={() => onChange(mode)}
            title={`切换到${meta.label}模式`}
            className={`flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[11px] font-medium transition ${
              active
                ? "bg-bg-elevated text-accent ring-1 ring-accent shadow-sm"
                : "text-ink-faint hover:text-ink-muted"
            }`}
          >
            <Icon className="w-3.5 h-3.5" />
            {meta.label}
          </button>
        );
      })}
    </div>
  );
}
