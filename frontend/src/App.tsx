import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  FileText,
  History,
  Inbox,
  Layers3,
  LoaderCircle,
  LogOut,
  Plus,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  SquareTerminal,
  Upload,
  X,
} from "lucide-react";

type Ticket = {
  id: string;
  title: string;
  customer: string;
  order_id: string | null;
  status: string;
  priority: string;
  description: string;
  version: number;
};
type Document = {
  id: string;
  name: string;
  source: string;
  chunks: number;
  updated_at: string;
};
type Evidence = {
  id: string;
  document_id: string;
  name: string;
  source: string;
  section: string;
  page: number | null;
  text: string;
  score: number;
};
type Trace = {
  step: number;
  tool: string;
  arguments: Record<string, unknown>;
  result: unknown;
  duration_ms: number;
  status: string;
};
type Proposal = {
  ticket_id: string;
  status: "in_progress" | "resolved";
  note: string;
  expected_version: number;
};
type Run = {
  id: string;
  question: string;
  ticket_id: string | null;
  mode: string;
  status:
    | "running"
    | "completed"
    | "awaiting_approval"
    | "rejected"
    | "failed"
    | "exhausted";
  answer: string;
  evidence: Evidence[];
  trace: Trace[];
  proposal: Proposal | null;
  usage: Record<string, number>;
  elapsed_ms: number;
  created_at: string;
};
type Page = "inbox" | "knowledge" | "history";
const labels: Record<string, string> = {
  open: "待处理",
  in_progress: "处理中",
  resolved: "已解决",
  closed: "已关闭",
  high: "高优先",
  normal: "普通",
  medium: "普通",
  low: "低优先",
  running: "处理中",
  completed: "已完成",
  awaiting_approval: "待审批",
  rejected: "已拒绝",
  failed: "失败",
  exhausted: "步骤已用尽",
  success: "成功",
  error: "错误",
  pending: "等待中",
};
const statusLabel = (value: string) => labels[value] || value;
const time = (value: string) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
};
function Badge({ status }: { status: string }) {
  return (
    <span className={`badge status-${status}`}>{statusLabel(status)}</span>
  );
}

export default function App() {
  const [token, setToken] = useState(
    () => sessionStorage.getItem("knowhub-token") || "",
  );
  const [draftToken, setDraftToken] = useState(token);
  const [connected, setConnected] = useState(false);
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [model, setModel] = useState("");
  const [page, setPage] = useState<Page>("inbox");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeRun, setActiveRun] = useState<Run | null>(null);
  const [tab, setTab] = useState<"sources" | "trace">("sources");
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<"agent" | "rag">("agent");
  const [search, setSearch] = useState("");
  const [busy, setBusy] = useState<string | null>(token ? "connect" : null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const questionInput = useRef<HTMLTextAreaElement>(null);
  const selected = tickets.find((t) => t.id === selectedId);

  async function request<T>(
    path: string,
    init: RequestInit = {},
    accessToken = token,
  ): Promise<T> {
    const response = await fetch(`/api${path}`, {
      ...init,
      headers: {
        Authorization: `Bearer ${accessToken}`,
        ...(init.body && !(init.body instanceof FormData)
          ? { "Content-Type": "application/json" }
          : {}),
        ...init.headers,
      },
    });
    const data: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const detail =
        data && typeof data === "object" && "detail" in data
          ? String(data.detail)
          : `请求失败（${response.status}）`;
      throw new Error(detail);
    }
    return data as T;
  }
  async function loadWorkspace(accessToken = token) {
    const [nextTickets, nextDocuments, nextRuns, health] = await Promise.all([
      request<Ticket[]>("/tickets", {}, accessToken),
      request<Document[]>("/documents", {}, accessToken),
      request<Run[]>("/runs", {}, accessToken),
      request<{ status: string; model: string }>("/health", {}, accessToken),
    ]);
    setTickets(nextTickets);
    setDocuments(nextDocuments);
    setRuns(nextRuns);
    setModel(health.model);
    setConnected(true);
  }
  async function connect(event?: FormEvent) {
    event?.preventDefault();
    const value = draftToken.trim();
    if (!value) return;
    setBusy("connect");
    setError("");
    try {
      await loadWorkspace(value);
      sessionStorage.setItem("knowhub-token", value);
      setToken(value);
      setNotice("工作台已连接");
    } catch (e) {
      setError(e instanceof Error ? e.message : "连接失败，请重试");
    } finally {
      setBusy(null);
    }
  }
  useEffect(() => {
    if (token) {
      loadWorkspace()
        .catch((e) => setError(e instanceof Error ? e.message : "加载失败"))
        .finally(() => setBusy(null));
    }
  }, []); // Initial session restoration only.
  async function refresh() {
    setBusy("refresh");
    setError("");
    try {
      await loadWorkspace();
      setNotice("数据已刷新");
    } catch (e) {
      setError(e instanceof Error ? e.message : "刷新失败");
    } finally {
      setBusy(null);
    }
  }
  function disconnect() {
    sessionStorage.removeItem("knowhub-token");
    setToken("");
    setDraftToken("");
    setConnected(false);
    setTickets([]);
    setDocuments([]);
    setRuns([]);
    setActiveRun(null);
    setSelectedId(null);
    setQuestion("");
    setError("");
    setNotice("");
  }
  function chooseTicket(ticket: Ticket) {
    setSelectedId(ticket.id);
    setActiveRun(null);
    setQuestion("");
    setTab("sources");
    setPage("inbox");
  }
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!question.trim() || busy) return;
    setBusy("run");
    setError("");
    setNotice("");
    setActiveRun(null);
    setTab("sources");
    try {
      const result = await request<Run>("/runs", {
        method: "POST",
        body: JSON.stringify({
          question: question.trim(),
          ticket_id: selectedId,
          mode,
        }),
      });
      setActiveRun(result);
      setRuns((previous) =>
        [result, ...previous.filter((r) => r.id !== result.id)].slice(0, 30),
      );
      setNotice(
        result.status === "awaiting_approval"
          ? "方案已生成，等待人工审批"
          : "处理已结束",
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "处理失败");
    } finally {
      setBusy(null);
    }
  }
  async function decide(approved: boolean) {
    if (!activeRun || busy || activeRun.status !== "awaiting_approval") return;
    setBusy("decision");
    setError("");
    try {
      const result = await request<Run>(
        `/runs/${encodeURIComponent(activeRun.id)}/decision`,
        { method: "POST", body: JSON.stringify({ approved }) },
      );
      setActiveRun(result);
      setRuns((previous) =>
        previous.map((r) => (r.id === result.id ? result : r)),
      );
      setNotice(approved ? "审批已提交，工单已更新" : "方案已拒绝，未执行写入");
      setTickets(await request<Ticket[]>("/tickets"));
    } catch (e) {
      setError(e instanceof Error ? e.message : "审批失败");
    } finally {
      setBusy(null);
    }
  }
  async function viewRun(run: Run) {
    setBusy("history");
    setError("");
    try {
      const result = await request<Run>(`/runs/${encodeURIComponent(run.id)}`);
      setActiveRun(result);
      setSelectedId(result.ticket_id);
      setQuestion(result.question);
      setMode(result.mode === "rag" ? "rag" : "agent");
      setPage("inbox");
      setTab("sources");
    } catch (e) {
      setError(e instanceof Error ? e.message : "读取记录失败");
    } finally {
      setBusy(null);
    }
  }
  async function upload(file?: File) {
    if (!file || busy) return;
    if (file.size > 5 * 1024 * 1024) {
      setError("文件超过 5 MB，请缩小后重新上传");
      return;
    }
    if (!/\.(pdf|md|txt)$/i.test(file.name)) {
      setError("请选择 PDF、MD 或 TXT 文件");
      return;
    }
    setBusy("upload");
    setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      const result = await request<{ document_id: string; chunks: number }>(
        "/documents/upload",
        { method: "POST", body },
      );
      setDocuments(await request<Document[]>("/documents"));
      setNotice(`已导入 ${file.name}，生成 ${result.chunks} 个知识片段`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "上传失败");
    } finally {
      setBusy(null);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  if (!connected)
    return (
      <div className="login-shell">
        <div className="login-art" aria-hidden="true">
          <div className="art-grid" />
          <div className="brand">
            <Layers3 size={27} />
            <span>
              KnowHub<span className="brand-period">.</span>
            </span>
          </div>
          <div className="login-statement">
            <span className="eyebrow">KNOWLEDGE INTO ACTION</span>
            <h1>
              让每次处理，
              <br />
              都有据可循。
            </h1>
            <p>
              连接知识与业务现场。
              <br />
              检索、判断、审批，在同一个工作台完成。
            </p>
          </div>
          <span className="login-foot">OPERATOR WORKSPACE / 01</span>
        </div>
        <main className="login-panel">
          <div className="mobile-brand">
            <Layers3 /> KnowHub.
          </div>
          <div className="login-form-wrap">
            <span className="eyebrow">运营工作台</span>
            <h2>连接你的工作空间</h2>
            <p className="muted">
              输入运营访问令牌，访问工单、企业知识与处理记录。
            </p>
            <form onSubmit={connect}>
              <label htmlFor="access-token">访问令牌</label>
              <input
                id="access-token"
                type="password"
                autoComplete="off"
                value={draftToken}
                onChange={(e) => setDraftToken(e.target.value)}
                placeholder="输入访问令牌"
                disabled={!!busy}
                required
              />
              {error && (
                <div role="alert" className="error-banner">
                  {error}
                </div>
              )}
              <button
                className="button primary login-submit"
                disabled={!!busy || !draftToken.trim()}
              >
                {busy ? (
                  <LoaderCircle className="spin" size={17} />
                ) : (
                  <ArrowRight size={17} />
                )}{" "}
                {busy ? "正在连接…" : "连接工作台"}
              </button>
            </form>
            <p className="token-note">
              <ShieldCheck size={16} />
              令牌仅保存在当前浏览器会话中
            </p>
          </div>
          <div className="login-bottom">
            KNOWHUB <span>知识驱动 · 人工掌控</span>
          </div>
        </main>
      </div>
    );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("inbox");
          }}
        >
          <Layers3 size={27} />
          <span>
            KnowHub<span className="brand-period">.</span>
          </span>
        </a>
        <div className="workspace-tag">
          <span className="workspace-avatar">K</span>
          <div>
            <strong>运营工作空间</strong>
            <span>共享企业知识库</span>
          </div>
          <ChevronDown size={15} />
        </div>
        <div className="nav-label">工作空间</div>
        <nav aria-label="主导航">
          {(
            [
              {
                key: "inbox",
                name: "工单工作台",
                icon: Inbox,
                count: tickets.length,
              },
              {
                key: "knowledge",
                name: "知识库",
                icon: BookOpen,
                count: documents.length,
              },
              {
                key: "history",
                name: "处理记录",
                icon: History,
                count: runs.length,
              },
            ] as const
          ).map((item) => (
            <button
              key={item.key}
              disabled={!!busy}
              className={`nav-item ${page === item.key ? "active" : ""}`}
              onClick={() => setPage(item.key)}
            >
              <item.icon size={18} />
              <span>{item.name}</span>
              <span className="nav-count">{item.count}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <ShieldCheck size={20} />
          <strong>每一步，都可核验</strong>
          <p>知识检索附带来源，业务写入须经人工审批。</p>
        </div>
        <div className="sidebar-bottom">
          <span className="connection-dot" />
          <span>已连接工作空间</span>
          <button
            title="退出当前会话"
            aria-label="退出当前会话"
            onClick={disconnect}
            disabled={!!busy}
          >
            <LogOut size={16} />
          </button>
        </div>
      </aside>
      <main className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            工作空间 <ChevronRight size={14} />{" "}
            <span>
              {page === "inbox"
                ? "工单工作台"
                : page === "knowledge"
                  ? "知识库"
                  : "处理记录"}
            </span>
          </div>
          <div className="topbar-actions">
            <span className="model-label" title={model}>
              <span className="connection-dot" />
              {model || "模型已连接"}
            </span>
            <button
              className="icon-button"
              aria-label="刷新工作空间"
              onClick={refresh}
              disabled={!!busy}
            >
              <RefreshCw
                size={16}
                className={busy === "refresh" ? "spin" : ""}
              />
            </button>
            <span className="operator-avatar" title="运营人员">
              OP
            </span>
          </div>
        </header>
        <div className="page-content">
          <div className="page-heading">
            <div>
              <span className="eyebrow">
                {page === "inbox"
                  ? "OPERATOR DESK"
                  : page === "knowledge"
                    ? "KNOWLEDGE LIBRARY"
                    : "ACTIVITY LOG"}
              </span>
              <h1>
                {page === "inbox"
                  ? "工单工作台"
                  : page === "knowledge"
                    ? "企业知识库"
                    : "处理记录"}
              </h1>
              <p>
                {page === "inbox"
                  ? "从问题到行动，让知识参与每一次业务决策。"
                  : page === "knowledge"
                    ? "将业务规则整理为可检索、可追溯的知识来源。"
                    : "回看问题、依据与操作，保留完整处理上下文。"}
              </p>
            </div>
            {page === "inbox" && (
              <button
                className="button outline"
                disabled={!!busy}
                onClick={() => {
                  setSelectedId(null);
                  setActiveRun(null);
                  setQuestion("");
                  questionInput.current?.focus();
                }}
              >
                <Plus size={16} />
                自由提问
              </button>
            )}
            {page === "knowledge" && (
              <button
                className="button primary"
                disabled={!!busy}
                onClick={() => fileInput.current?.click()}
              >
                {busy === "upload" ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <Upload size={16} />
                )}{" "}
                {busy === "upload" ? "正在入库…" : "上传知识"}
              </button>
            )}
          </div>
          {error && (
            <div className="error-banner global-banner" role="alert">
              <CircleHelp size={17} />
              <span>{error}</span>
              <button aria-label="关闭错误提示" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          <div className="sr-only" role="status" aria-live="polite">
            {busy
              ? {
                  run: "正在处理请求，请稍候",
                  upload: "正在上传并构建知识索引，请稍候",
                  decision: "正在提交审批结果",
                  refresh: "正在刷新工作空间",
                  history: "正在读取处理记录",
                  connect: "正在连接工作空间",
                }[busy] || "正在处理"
              : notice}
          </div>
          {notice && (
            <div className="notice">
              <Check size={14} />
              {notice}
            </div>
          )}
          {page === "inbox" && (
            <div className="workbench">
              <section className="inbox-panel" aria-label="工单列表">
                <div className="panel-title">
                  <h2>
                    收件箱 <span>{tickets.length}</span>
                  </h2>
                  <span className="tiny-label">全部工单</span>
                </div>
                <label className="search-field">
                  <Search size={16} />
                  <input
                    aria-label="搜索工单"
                    placeholder="搜索工单、客户…"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                  <span>⌕</span>
                </label>
                <div className="ticket-list">
                  {tickets
                    .filter((t) =>
                      `${t.id} ${t.title} ${t.customer}`
                        .toLowerCase()
                        .includes(search.toLowerCase()),
                    )
                    .map((ticket) => (
                      <button
                        disabled={!!busy}
                        key={ticket.id}
                        className={`ticket-card ${selectedId === ticket.id ? "selected" : ""}`}
                        onClick={() => chooseTicket(ticket)}
                      >
                        <div className="ticket-meta">
                          <span>{ticket.id}</span>
                          <Badge status={ticket.status} />
                        </div>
                        <h3>{ticket.title}</h3>
                        <p>{ticket.description}</p>
                        <div className="ticket-footer">
                          <span className="customer-avatar">
                            {ticket.customer.slice(0, 1)}
                          </span>
                          <span>{ticket.customer}</span>
                          <span
                            className={`priority priority-${ticket.priority}`}
                          >
                            <i />
                            {statusLabel(ticket.priority)}
                          </span>
                        </div>
                      </button>
                    ))}
                  {tickets.length === 0 && (
                    <div className="small-empty">
                      暂无工单，可以直接向知识库提问。
                    </div>
                  )}
                  {tickets.length > 0 &&
                    !tickets.some((t) =>
                      `${t.id} ${t.title} ${t.customer}`
                        .toLowerCase()
                        .includes(search.toLowerCase()),
                    ) && <div className="small-empty">没有匹配的工单</div>}
                </div>
                <div className="inbox-foot">
                  <span className="connection-dot" />
                  演示客户与订单均为合成数据
                </div>
              </section>
              <section className="agent-panel" aria-label="智能处理工作区">
                <div className="context-bar">
                  <div className="context-icon">
                    {selected ? <Inbox size={18} /> : <Sparkles size={18} />}
                  </div>
                  <div>
                    <strong>
                      {selected ? selected.title : "知识与业务助手"}
                    </strong>
                    <span>
                      {selected
                        ? `${selected.id} · ${selected.customer}${selected.order_id ? ` · ${selected.order_id}` : ""}`
                        : "选择左侧工单，或开始一次知识问答"}
                    </span>
                  </div>
                  {selected && <Badge status={selected.status} />}
                </div>
                <div className="agent-content">
                  {!activeRun && busy !== "run" && (
                    <div className="welcome">
                      <div className="welcome-icon">
                        <Layers3 size={29} />
                      </div>
                      <span className="eyebrow">
                        GROUNDED IN YOUR KNOWLEDGE
                      </span>
                      <h2>
                        {selected
                          ? "为这张工单，找到有依据的下一步。"
                          : "让知识，成为下一步行动。"}
                      </h2>
                      <p>
                        {selected
                          ? selected.description
                          : "检索企业规则，核实业务状态，并在你的审批后执行更新。"}
                      </p>
                      <div className="capability-row">
                        <span>
                          <BookOpen size={14} />
                          可追溯的来源
                        </span>
                        <span>
                          <SquareTerminal size={14} />
                          真实工具记录
                        </span>
                        <span>
                          <ShieldCheck size={14} />
                          人工审批写入
                        </span>
                      </div>
                      <div className="suggestions">
                        <span className="tiny-label">试着提问</span>
                        {[
                          selected
                            ? `查询 ${selected.id} 的订单状态，按知识库规则提出处理方案，并准备将工单改为处理中。`
                            : "有哪些退款处理规则？",
                          "请按公司火星运输赔付政策处理。",
                        ].map((prompt) => (
                          <button
                            key={prompt}
                            onClick={() => {
                              setQuestion(prompt);
                              questionInput.current?.focus();
                            }}
                            disabled={!!busy}
                          >
                            {prompt}
                            <ArrowRight size={15} />
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {busy === "run" && (
                    <div className="working-state" role="status">
                      <div className="working-orbit">
                        <LoaderCircle size={30} className="spin" />
                      </div>
                      <h2>
                        {mode === "agent"
                          ? "正在核实并处理请求"
                          : "正在检索知识库"}
                      </h2>
                      <p>
                        等待模型与工具返回真实结果。完成后会展示答案、来源和执行记录。
                      </p>
                      <span className="working-question">{question}</span>
                      <span className="tiny-label">请保持此页面开启</span>
                    </div>
                  )}
                  {activeRun && (
                    <div className="run-result">
                      <div className="answer-heading">
                        <div className="answer-avatar">
                          <Sparkles size={17} />
                        </div>
                        <strong>处理结果</strong>
                        <Badge status={activeRun.status} />
                        <span>
                          {(activeRun.elapsed_ms / 1000).toFixed(1)} 秒
                        </span>
                      </div>
                      <p className="answer-text">
                        {activeRun.answer ||
                          "本次处理未返回答案，请检查执行记录。"}
                      </p>
                      {(activeRun.status === "failed" ||
                        activeRun.status === "exhausted") && (
                        <div className="result-warning">
                          本次处理未完整完成。请查看执行轨迹，核实信息后再尝试。
                        </div>
                      )}
                      {activeRun.proposal && (
                        <div
                          className={`approval-card ${activeRun.status === "awaiting_approval" ? "pending" : ""}`}
                        >
                          <div className="approval-header">
                            <ShieldCheck size={20} />
                            <div>
                              <strong>
                                {activeRun.status === "awaiting_approval"
                                  ? "有一项操作等待你的审批"
                                  : activeRun.status === "rejected"
                                    ? "操作方案已拒绝"
                                    : "操作方案"}
                              </strong>
                              <p>
                                将更新工单 {activeRun.proposal.ticket_id}{" "}
                                的状态与处理备注
                              </p>
                            </div>
                          </div>
                          <div className="proposal-diff">
                            <div>
                              <span>工单状态</span>
                              <strong>
                                {statusLabel(activeRun.proposal.status)}
                              </strong>
                            </div>
                            <div>
                              <span>处理备注</span>
                              <p>{activeRun.proposal.note}</p>
                            </div>
                            <div>
                              <span>版本校验</span>
                              <code>
                                v{activeRun.proposal.expected_version}
                              </code>
                            </div>
                          </div>
                          {activeRun.status === "awaiting_approval" ? (
                            <div className="approval-footer">
                              <span>批准后写入业务系统</span>
                              <button
                                className="button subtle"
                                disabled={!!busy}
                                onClick={() => decide(false)}
                              >
                                <X size={15} />
                                拒绝
                              </button>
                              <button
                                className="button primary"
                                disabled={!!busy}
                                onClick={() => decide(true)}
                              >
                                {busy === "decision" ? (
                                  <LoaderCircle className="spin" size={15} />
                                ) : (
                                  <Check size={15} />
                                )}
                                批准并写入
                              </button>
                            </div>
                          ) : (
                            <div className="approval-resolved">
                              {activeRun.status === "rejected" ? (
                                <X size={15} />
                              ) : (
                                <Check size={15} />
                              )}{" "}
                              {activeRun.status === "rejected"
                                ? "已拒绝 · 未执行该方案"
                                : activeRun.status === "completed"
                                  ? "审批流程已完成"
                                  : "请结合运行状态核实执行结果"}
                            </div>
                          )}
                        </div>
                      )}
                      <div
                        className="result-tabs"
                        role="tablist"
                        aria-label="结果详情"
                      >
                        <button
                          role="tab"
                          aria-selected={tab === "sources"}
                          onClick={() => setTab("sources")}
                        >
                          <BookOpen size={15} />
                          参考来源 <span>{activeRun.evidence.length}</span>
                        </button>
                        <button
                          role="tab"
                          aria-selected={tab === "trace"}
                          onClick={() => setTab("trace")}
                        >
                          <SquareTerminal size={15} />
                          执行轨迹 <span>{activeRun.trace.length}</span>
                        </button>
                      </div>
                      {tab === "sources" ? (
                        <div
                          className="sources"
                          role="tabpanel"
                          aria-label="参考来源"
                        >
                          {activeRun.evidence.map((source, index) => (
                            <SourceCard
                              key={`${source.id}-${index}`}
                              source={source}
                              index={index}
                            />
                          ))}
                          {!activeRun.evidence.length && (
                            <div className="small-empty">
                              本次没有检索到可引用的知识来源。请审慎核实答案。
                            </div>
                          )}
                        </div>
                      ) : (
                        <div
                          className="trace-list"
                          role="tabpanel"
                          aria-label="执行轨迹"
                        >
                          {activeRun.trace.map((step, index) => (
                            <TraceCard
                              key={`${step.step}-${index}`}
                              trace={step}
                            />
                          ))}
                          {!activeRun.trace.length && (
                            <div className="small-empty">
                              本次未产生工具调用记录。
                            </div>
                          )}
                          <div className="run-id">运行 ID · {activeRun.id}</div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                <form className="composer" onSubmit={submit}>
                  <div className="composer-top">
                    <span className="composer-label">
                      {selected ? (
                        <>
                          <span className="connection-dot" />
                          {selected.id} 作为业务上下文
                        </>
                      ) : (
                        <>
                          <BookOpen size={13} />
                          面向整个知识库提问
                        </>
                      )}
                    </span>
                    <div className="mode-switch" aria-label="处理模式">
                      <button
                        type="button"
                        disabled={!!busy}
                        aria-pressed={mode === "agent"}
                        onClick={() => setMode("agent")}
                      >
                        Agent
                      </button>
                      <button
                        type="button"
                        disabled={!!busy}
                        aria-pressed={mode === "rag"}
                        onClick={() => setMode("rag")}
                      >
                        仅检索
                      </button>
                    </div>
                  </div>
                  <textarea
                    ref={questionInput}
                    aria-label="向知识库提问"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    disabled={!!busy}
                    placeholder={
                      selected
                        ? "描述处理目标，例如：核实订单状态，并准备更新工单…"
                        : "输入业务问题，检索规则或请求处理…"
                    }
                    rows={3}
                    required
                  />
                  <div className="composer-bottom">
                    <span>
                      {mode === "agent"
                        ? "可查询业务工具；写入操作须审批"
                        : "仅检索知识并回答，不执行业务操作"}
                    </span>
                    <button
                      className="button primary"
                      disabled={!!busy || !question.trim()}
                      type="submit"
                    >
                      {busy === "run" ? (
                        <LoaderCircle size={15} className="spin" />
                      ) : (
                        <Send size={15} />
                      )}{" "}
                      {busy === "run" ? "处理中…" : "开始处理"}
                    </button>
                  </div>
                </form>
                <div className="agent-disclaimer">
                  答案基于可用知识与工具返回，请核实来源后作出业务判断。
                </div>
              </section>
            </div>
          )}
          {page === "knowledge" && (
            <section className="library-panel">
              <div className="library-intro">
                <div className="intro-icon">
                  <BookOpen size={25} />
                </div>
                <div>
                  <h2>业务规则的共同底稿</h2>
                  <p>
                    支持 PDF、Markdown 与纯文本，每个文件不超过 5
                    MB。上传后自动整理为可检索的知识片段。
                  </p>
                </div>
                <button
                  className="button outline"
                  disabled={!!busy}
                  onClick={() => fileInput.current?.click()}
                >
                  <ArrowDownToLine size={15} />
                  选择文件
                </button>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>文档名称</th>
                      <th>来源</th>
                      <th>知识片段</th>
                      <th>更新时间</th>
                    </tr>
                  </thead>
                  <tbody>
                    {documents.map((doc) => (
                      <tr key={doc.id}>
                        <td>
                          <div className="document-name">
                            <FileText size={19} />
                            <strong>{doc.name}</strong>
                          </div>
                        </td>
                        <td className="source-path">{doc.source}</td>
                        <td>
                          <span className="chunk-count">
                            {doc.chunks} 个片段
                          </span>
                        </td>
                        <td>{time(doc.updated_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {!documents.length && (
                <div className="empty-library">
                  <Upload size={30} />
                  <h3>从第一份业务规则开始</h3>
                  <p>上传真实文档，让回答有据可循。</p>
                  <button
                    className="button primary"
                    onClick={() => fileInput.current?.click()}
                    disabled={!!busy}
                  >
                    上传文档
                  </button>
                </div>
              )}
              <div className="table-foot">
                {documents.length} 份文档 · 仅展示已入库的知识来源
              </div>
            </section>
          )}
          <input
            className="sr-only"
            tabIndex={-1}
            ref={fileInput}
            type="file"
            aria-label="上传知识文件"
            accept=".pdf,.md,.txt"
            disabled={!!busy}
            onChange={(e) => {
              void upload(e.target.files?.[0]);
            }}
          />
          {page === "history" && (
            <section className="history-panel">
              <div className="panel-title">
                <h2>最近的处理记录</h2>
                <span className="tiny-label">最多 30 条 · 服务端保存</span>
              </div>
              <div className="history-list">
                {runs.map((run) => (
                  <button
                    className="history-row"
                    key={run.id}
                    disabled={!!busy}
                    onClick={() => viewRun(run)}
                  >
                    <div className="history-icon">
                      {run.mode === "agent" ? (
                        <Sparkles size={18} />
                      ) : (
                        <Search size={18} />
                      )}
                    </div>
                    <div className="history-main">
                      <strong>{run.question}</strong>
                      <span>
                        {run.ticket_id || "知识库问答"}
                        <i /> {time(run.created_at)}
                        <i />
                        {run.mode === "agent" ? "Agent" : "仅检索"}
                      </span>
                    </div>
                    <Badge status={run.status} />
                    <span className="history-duration">
                      <Clock3 size={13} />
                      {(run.elapsed_ms / 1000).toFixed(1)} 秒
                    </span>
                    <ChevronRight size={17} />
                  </button>
                ))}
              </div>
              {!runs.length && (
                <div className="empty-library">
                  <History size={30} />
                  <h3>还没有处理记录</h3>
                  <p>完成一次提问后，可在这里回看结果与执行过程。</p>
                  <button
                    className="button outline"
                    onClick={() => setPage("inbox")}
                  >
                    前往工作台
                    <ArrowRight size={15} />
                  </button>
                </div>
              )}
            </section>
          )}
          <footer className="page-footer">
            <span>KnowHub / 企业知识行动台</span>
            <span>检索有据 · 执行有序</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
function SourceCard({ source, index }: { source: Evidence; index: number }) {
  return (
    <article className="source-card">
      <header>
        <span className="source-index">
          {String(index + 1).padStart(2, "0")}
        </span>
        <FileText size={16} />
        <strong>{source.name}</strong>
        <span className="source-location">
          {source.page !== null
            ? `第 ${source.page} 页`
            : source.section || "知识片段"}
        </span>
      </header>
      <blockquote>{source.text}</blockquote>
      <footer>
        <span>{source.source}</span>
        {source.section && <span>{source.section}</span>}
      </footer>
    </article>
  );
}
function TraceCard({ trace }: { trace: Trace }) {
  return (
    <details className="trace-card">
      <summary>
        <span className="trace-number">
          {String(trace.step).padStart(2, "0")}
        </span>
        <SquareTerminal size={16} />
        <strong>{trace.tool}</strong>
        <Badge status={trace.status} />
        <span className="trace-time">{trace.duration_ms} ms</span>
        <ChevronDown size={15} />
      </summary>
      <div className="trace-detail">
        <span className="tiny-label">调用参数</span>
        <pre>{JSON.stringify(trace.arguments, null, 2)}</pre>
        <span className="tiny-label">工具返回</span>
        <pre>{JSON.stringify(trace.result, null, 2)}</pre>
      </div>
    </details>
  );
}
