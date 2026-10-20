import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
const ticket = {
  id: "TK-1042",
  title: "订单配送延迟",
  customer: "示例客户",
  order_id: "ORD-1001",
  status: "open",
  priority: "high",
  description: "请查询订单",
  version: 1,
};
const run = {
  id: "RUN-1",
  question: "处理工单",
  ticket_id: "TK-1042",
  mode: "agent",
  status: "awaiting_approval",
  answer: "根据规则准备处理。",
  evidence: [
    {
      id: "e1",
      document_id: "d1",
      name: "配送规则",
      source: "policy.md",
      section: "延迟",
      page: null,
      text: "延迟订单需核实。",
      score: 0.8,
    },
  ],
  trace: [
    {
      step: 1,
      tool: "get_order",
      arguments: { order_id: "ORD-1001" },
      result: { status: "delayed" },
      duration_ms: 24,
      status: "completed",
    },
  ],
  proposal: {
    ticket_id: "TK-1042",
    status: "in_progress",
    note: "已核实物流",
    expected_version: 1,
  },
  usage: {},
  elapsed_ms: 250,
  created_at: "2026-10-20T00:00:00Z",
};
beforeEach(() => {
  sessionStorage.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      if (url === "/api/health")
        return Response.json({ status: "ok", model: "test-model" });
      if (url === "/api/tickets") return Response.json([ticket]);
      if (url === "/api/documents") return Response.json([]);
      if (url === "/api/runs" && init?.method === "POST")
        return Response.json(run);
      if (url === "/api/runs/RUN-1/decision")
        return Response.json({
          ...run,
          status: "completed",
          proposal: run.proposal,
        });
      return Response.json([]);
    }),
  );
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
describe("operator workspace", () => {
  it("stores the session token and shows authenticated inbox", async () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText("访问令牌"), {
      target: { value: "local-test" },
    });
    fireEvent.click(screen.getByRole("button", { name: "连接工作台" }));
    await screen.findByText("订单配送延迟");
    expect(sessionStorage.getItem("knowhub-token")).toBe("local-test");
    expect(fetch).toHaveBeenCalledWith(
      "/api/tickets",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer local-test",
        }),
      }),
    );
  });
  it("shows source and public tool evidence and approves a real proposal", async () => {
    sessionStorage.setItem("knowhub-token", "local-test");
    render(<App />);
    await screen.findByText("订单配送延迟");
    fireEvent.click(screen.getByRole("button", { name: /订单配送延迟/ }));
    fireEvent.change(screen.getByLabelText("向知识库提问"), {
      target: { value: "处理工单" },
    });
    fireEvent.click(screen.getByRole("button", { name: "开始处理" }));
    await screen.findByText("根据规则准备处理。");
    expect(screen.getByText("延迟订单需核实。")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: /执行轨迹/ }));
    expect(screen.getByText("get_order")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "批准并写入" }));
    await waitFor(() =>
      expect(fetch).toHaveBeenCalledWith(
        "/api/runs/RUN-1/decision",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ approved: true }),
        }),
      ),
    );
  });
  it("renders API authorization failures without pretending to load data", async () => {
    sessionStorage.setItem("knowhub-token", "bad-token");
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify({ detail: "令牌无效" }), { status: 401 }),
      ),
    );
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveProperty(
      "textContent",
      expect.stringContaining("令牌无效"),
    );
  });
  it("keeps a real waiting state and disables conflicting controls until the synchronous run returns", async () => {
    sessionStorage.setItem("knowhub-token", "local-test");
    const originalFetch = fetch;
    let finish: ((value: Response) => void) | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string, init?: RequestInit) =>
        url === "/api/runs" && init?.method === "POST"
          ? new Promise<Response>((resolve) => {
              finish = resolve;
            })
          : originalFetch(url, init),
      ),
    );
    render(<App />);
    await screen.findByText("订单配送延迟");
    fireEvent.change(screen.getByLabelText("向知识库提问"), {
      target: { value: "核实规则" },
    });
    fireEvent.click(screen.getByRole("button", { name: "开始处理" }));
    expect(await screen.findByText("正在核实并处理请求")).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: "仅检索" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(screen.queryByText("get_order")).toBeNull();
    finish?.(Response.json(run));
    await screen.findByText("根据规则准备处理。");
  });
  it("rejects unsupported knowledge files before sending an upload", async () => {
    sessionStorage.setItem("knowhub-token", "local-test");
    render(<App />);
    await screen.findByText("订单配送延迟");
    const file = new File(["invalid"], "script.exe");
    fireEvent.change(screen.getByLabelText("上传知识文件"), {
      target: { files: [file] },
    });
    expect(await screen.findByRole("alert")).toHaveProperty(
      "textContent",
      expect.stringContaining("请选择 PDF、MD 或 TXT 文件"),
    );
    expect(fetch).not.toHaveBeenCalledWith(
      "/api/documents/upload",
      expect.anything(),
    );
  });
});
