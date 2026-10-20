# Operator API contract

Base /api. All routes except GET /health require Authorization: Bearer <operator token>. Single shared authorized workspace. JSON errors use {detail:string}. Arrays are returned directly (not wrapped).

- GET /health -> {status:'ok', model:string}
- GET /tickets -> Ticket[]
- GET /documents -> Document[]
- POST /documents/upload multipart field file (PDF/MD/TXT, max5MB) -> {document_id:string,chunks:number}
- POST /runs JSON {question:string,ticket_id?:string|null,mode?:'agent'|'rag'} -> Run (synchronous; display a real waiting state while processing, no fake intermediate trace).
- GET /runs -> Run[] (newest first, max30)
- GET /runs/{id} -> Run
- POST /runs/{id}/decision JSON {approved:boolean} -> Run

```typescript
type Ticket = {id:string; title:string; customer:string; order_id:string|null; status:string; priority:string; description:string; version:number};
type Document = {id:string; name:string; source:string; chunks:number; updated_at:string};
type Evidence = {id:string; document_id:string; name:string; source:string; section:string; page:number|null; text:string; score:number};
type Trace = {step:number; tool:string; arguments:Record<string,unknown>; result:unknown; duration_ms:number; status:string};
type Proposal = {ticket_id:string; status:'in_progress'|'resolved'; note:string; expected_version:number};
type Run = {id:string; question:string; ticket_id:string|null; mode:string; status:'running'|'completed'|'awaiting_approval'|'rejected'|'failed'|'exhausted'; answer:string; evidence:Evidence[]; trace:Trace[]; proposal:Proposal|null; usage:Record<string,number>; elapsed_ms:number; created_at:string};
```

Demo TK-1042 concerns delayed order ORD-1001 (synthetic). TK-1043 concerns refunds for ORD-1002. TK-1044 is an unknown custom compliance request with no policy, to demonstrate uncertainty. Demo customer/ticket/order values are deliberately synthetic. Standard questions: “查询 TK-1042 的订单状态，按知识库规则提出处理方案，并准备将工单改为处理中。” / “有哪些退款处理规则？” / “请按公司火星运输赔付政策处理。”

UI should allow input/editing token, browsing tickets/documents, uploading, asking with selected ticket, showing source passages (render text safely), inspecting actual public tool traces and recent runs, approving/rejecting proposals. API errors must be visible. Do not expose raw model thinking. Do not display numeric success/latency claims without actual response data. No public deployment implied.
