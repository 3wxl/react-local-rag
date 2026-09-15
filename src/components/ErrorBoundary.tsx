import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

/**
 * 全局 ErrorBoundary：捕获组件渲染期未处理的异常，
 * 展示友好错误页而非白屏。
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("未捕获的渲染异常", error, info?.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-bg px-4 text-center">
          <div className="max-w-md">
            <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-red-500/15 flex items-center justify-center text-red-500 text-2xl">
              !
            </div>
            <h1 className="text-lg font-semibold text-ink mb-2">页面出了点问题</h1>
            <p className="text-sm text-ink-muted mb-1">
              {this.state.error?.message || "发生了未知错误"}
            </p>
            <p className="text-xs text-ink-faint mb-6">
              刷新页面通常可以恢复正常。如果反复出现，请检查浏览器控制台获取详情。
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:opacity-90 transition"
            >
              刷新页面
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
