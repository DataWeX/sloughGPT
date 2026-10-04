import{r as e}from"./rolldown-runtime-hePW80VL.js";import{t}from"./react-Cvdyeg_0.js";import{t as n}from"./jsx-runtime-NZYk81nU.js";import{n as r,o as i,r as a,s as o}from"./skeleton-BbdGmVPN.js";import{t as s}from"./progress-BtYaQKPk.js";import{i as c,n as l,r as ee,t as te}from"./tabs-kEWdJi45.js";import{t as ne}from"./PageContainer-V_2Lp6MF.js";import{t as u}from"./error-utils-CYINnpz6.js";import{t as re}from"./V86TerminalPanel-DORCOv-o.js";import{t as d}from"./db-_Yer2z8J.js";import{Dr as f,Er as p,Ut as ie,fr as ae,tt as m}from"./index-Ck7b__zA.js";import{t as h}from"./StatusBanner-CsZLYPMS.js";import{t as oe}from"./useRefreshShortcut-BF9tZFpx.js";import{t as se}from"./constants-Dx6jfxoK.js";var g=e(t()),_=n(),v=5e3,y=1e6,b={dataset:`shakespeare`,epochs:1,lr:.001,batch_size:32,n_layer:4,n_head:4,embed_dim:128};function x(e){return`[BITS 32]

; Launch a training job through the x86 training bridge.
; SYS_TRAIN_START: EAX=28, EBX=config JSON addr.
; Returns EAX = job_id (>=1), or -2 when permission denied.
; Requires the ADMIN role (role selector above).
; The Training card on the right polls the job to completion, shows the
; final result, and offers a Stop button for running jobs.

MOV EBX, config
MOV EAX, 28
INT 0x80
HLT

config: db '${JSON.stringify(e)}', 0`}function S(e){return Number.isFinite(e)?Math.min(y,Math.max(1,Math.floor(e))):v}async function ce(){try{let e=await d.getKV(`vm-role`)??`user`;if(e===`admin`||e===`kernel`||e===`user`)return e}catch{}return`user`}async function le(){try{let e=await d.getKV(`vm-max-steps`),t=Number(e??NaN);if(Number.isFinite(t))return S(t)}catch{}return v}async function ue(){try{let e=await d.getKV(`vm-train-config`);if(e)return C(e)}catch{}return b}function C(e){return{...e,dataset:e.dataset.trim()||b.dataset,epochs:Number.isFinite(e.epochs)?Math.max(1,Math.floor(e.epochs)):b.epochs,lr:Number.isFinite(e.lr)&&e.lr>0?e.lr:b.lr,batch_size:Number.isFinite(e.batch_size)?Math.max(1,Math.floor(e.batch_size)):b.batch_size,n_layer:Number.isFinite(e.n_layer)?Math.max(1,Math.floor(e.n_layer)):b.n_layer,n_head:Number.isFinite(e.n_head)?Math.max(1,Math.floor(e.n_head)):b.n_head,embed_dim:Number.isFinite(e.embed_dim)?Math.max(1,Math.floor(e.embed_dim)):b.embed_dim}}var w=e=>e.trim()===``?NaN:Number(e);function T(e){let t=[];return e.dataset.trim()||t.push({label:`Dataset`,message:`using default "${b.dataset}"`}),Number.isFinite(e.epochs)||t.push({label:`Epochs`,message:`using default ${b.epochs}`}),(!Number.isFinite(e.lr)||e.lr<=0)&&t.push({label:`Learning rate`,message:`using default ${b.lr}`}),Number.isFinite(e.batch_size)||t.push({label:`Batch size`,message:`using default ${b.batch_size}`}),Number.isFinite(e.n_layer)||t.push({label:`Layers`,message:`using default ${b.n_layer}`}),Number.isFinite(e.n_head)||t.push({label:`Heads`,message:`using default ${b.n_head}`}),Number.isFinite(e.embed_dim)||t.push({label:`Embed size`,message:`using default ${b.embed_dim}`}),t}var E={hello:`[BITS 32]

; Write "Hello, VM!" to VGA text buffer
MOV ESI, msg
MOV EDI, 0xB8000
MOV AH, 0x07

.loop:
LODSB
OR AL, AL
JZ .done
STOSW
JMP .loop

.done:
HLT

msg: db 'Hello, VM!', 0`,count:`[BITS 32]

; Count 0-9, write digits to VGA
MOV ECX, 0
MOV EDI, 0xB8000
MOV AH, 0x0B

.loop:
CMP ECX, 10
JGE .done
MOV AL, CL
ADD AL, 48
STOSW
INC ECX
JMP .loop

.done:
HLT`,fib:`[BITS 32]

; Fibonacci: compute and display first 10 numbers on VGA
MOV EAX, 0
MOV EBX, 1
MOV ECX, 0
MOV EDI, 0xB8000

.loop:
CMP ECX, 10
JGE .done

; Convert EAX (fib value) to ASCII
PUSH EAX
PUSH EBX
PUSH ECX
MOV EBX, 10
XOR ECX, 0

.digit_loop:
XOR EDX, EDX
DIV EBX
ADD DL, 48
PUSH EDX
INC ECX
TEST EAX, EAX
JNZ .digit_loop

; Write digits to VGA
MOV AH, 0x0A
.write_loop:
POP EDX
MOV AL, DL
STOSW
LOOP .write_loop

; Write space
MOV WORD [ES:EDI], 0x0A20
ADD EDI, 2

POP ECX
POP EBX
POP EAX

; next = fib(n-2) + fib(n-1)
MOV EDX, EAX
ADD EDX, EBX
MOV EAX, EBX
MOV EBX, EDX

INC ECX
JMP .loop

.done:
HLT`,sort:`[BITS 32]

; Bubble sort 8 bytes, display sorted array on VGA
MOV ECX, 7

.outer:
MOV EDX, 0
MOV BYTE [swapped], 0

.inner:
MOV AL, [arr + EDX]
MOV BL, [arr + EDX + 1]
CMP AL, BL
JLE .no_swap
MOV [arr + EDX], BL
MOV [arr + EDX + 1], AL
MOV BYTE [swapped], 1

.no_swap:
INC EDX
CMP EDX, ECX
JL .inner

CMP BYTE [swapped], 0
JZ .done
DEC ECX
JMP .outer

.done:
; Display sorted array on VGA
MOV ESI, arr
MOV EDI, 0xB8000
MOV AH, 0x0E

MOV ECX, 8
.disp_loop:
LODSB
ADD AL, 48
STOSW
LOOP .disp_loop

HLT

arr: db 5, 3, 8, 1, 9, 2, 7, 4
swapped: db 0`,vga_color:`[BITS 32]

; Rainbow stripe pattern on VGA
MOV EDI, 0xB8000
MOV ECX, 25
MOV BL, 1

.row_loop:
PUSH ECX
MOV ECX, 80

.col_loop:
MOV AL, '*'
MOV AH, BL
STOSW
LOOP .col_loop

POP ECX
INC BL
CMP BL, 16
JL .no_wrap
MOV BL, 1

.no_wrap:
LOOP .row_loop
HLT`,rainbow:`[BITS 32]

; Rainbow text "HELLO VM!" on VGA
MOV ESI, msg
MOV EDI, 0xB8000
ADD EDI, 160

MOV ECX, 9
MOV EBX, colors

.loop:
LODSB
MOV AH, [EBX]
INC EBX
STOSW
LOOP .loop

HLT

msg: db 'HELLO VM!'
colors: db 4, 14, 2, 1, 5, 6, 3, 11, 4`,primes:`[BITS 32]

; Sieve of Eratosthenes — find primes up to 50, display on VGA
MOV EDI, sieve
MOV ECX, 51
MOV AL, 1
REP STOSB

MOV ESI, 2

.sieve_loop:
CMP ESI, 51
JGE .sieve_done
MOV AL, [sieve + ESI]
TEST AL, AL
JZ .next
MOV EDI, ESI
ADD EDI, ESI

.mark_loop:
CMP EDI, 51
JGE .next
MOV BYTE [sieve + EDI], 0
ADD EDI, ESI
JMP .mark_loop

.next:
INC ESI
JMP .sieve_loop

.sieve_done:
; Display primes on VGA
MOV ESI, 2
MOV EDI, 0xB8000
MOV AH, 0x0B

.print_loop:
CMP ESI, 51
JGE .done
MOV AL, [sieve + ESI]
TEST AL, AL
JZ .skip
MOV AL, SIL
ADD AL, 48
STOSW
; space
MOV WORD [ES:EDI], 0x0B20
ADD EDI, 2

.skip:
INC ESI
JMP .print_loop

.done:
HLT

sieve: times 51 db 0`,calculator:`[BITS 32]

; Compute 7 * 8 + 5 = 61, display on VGA
MOV EAX, 7
MOV EBX, 8
MUL EBX
ADD EAX, 5

; Convert EAX to decimal (2 digits)
MOV EBX, 10
XOR EDX, EDX
DIV EBX
ADD AL, 48
ADD DL, 48
MOV [result], AL
MOV [result+1], DL
MOV BYTE [result+2], 0

; Write to VGA
MOV ESI, result
MOV EDI, 0xB8000
MOV AH, 0x0A

.loop:
LODSB
OR AL, AL
JZ .done
STOSW
JMP .loop

.done:
HLT

result: times 4 db 0`,factorial:`[BITS 32]

; Compute 6! = 720, display on VGA
MOV EAX, 1
MOV ECX, 6

.loop:
MUL ECX
DEC ECX
JNZ .loop

; EAX = 720, convert to decimal
MOV EBX, 10
MOV ECX, 0

.digit_loop:
XOR EDX, EDX
DIV EBX
ADD DL, 48
PUSH EDX
INC ECX
TEST EAX, EAX
JNZ .digit_loop

; Pop digits to VGA
MOV EDI, 0xB8000
MOV AH, 0x0E

.write_loop:
POP EDX
MOV AL, DL
STOSW
LOOP .write_loop

HLT`,guess:`[BITS 32]

; Number guessing game (pre-set answer = 5)
MOV ESI, prompt
MOV EDI, 0xB8000
MOV AH, 0x0A

.prompt_loop:
LODSB
OR AL, AL
JZ .done
STOSW
JMP .prompt_loop

.done:
HLT

prompt: db 'Answer is 5!', 0`,train:`[BITS 32]

; Launch a training job through the x86 training bridge.
; SYS_TRAIN_START: EAX=28, EBX=config JSON addr.
; Returns EAX = job_id (>=1), or -2 when permission denied.
; Requires the ADMIN role (role selector above).
; The Training card on the right polls the job to completion, shows the
; final result, and offers a Stop button for running jobs.

MOV EBX, config
MOV EAX, 28
INT 0x80
HLT

config: db '{"dataset":"shakespeare","epochs":1}', 0`,"train-status":`[BITS 32]

; Inspect a training job via the x86 training bridge.
; Requires the ADMIN role (role selector above).
; 1) SYS_TRAIN_STATUS: EAX=29, EBX=job_id
;    EAX: 0 running | 1 completed | 2 failed | -1 not found
; 2) SYS_TRAIN_GET_RESULT: EAX=30, EBX=job_id, ECX=buf, EDX=size
;    Writes result JSON into buf, returns bytes written in EAX.
; Both results are stored to guest memory; the returned JSON is also
; surfaced in the Training result card on the right.

MOV EBX, 1
MOV EAX, 29
INT 0x80
MOV [0x5000], EAX

MOV EBX, 1
MOV EAX, 30
MOV ECX, 0x90000
MOV EDX, 256
INT 0x80
MOV [0x5004], EAX
HLT`};function D(){let[e,t]=(0,g.useState)(`assembly`),[n,s]=(0,g.useState)(E.hello),[p,D]=(0,g.useState)(null),[A,j]=(0,g.useState)(!1),[M,N]=(0,g.useState)(v),[P,fe]=(0,g.useState)(!1),[F,pe]=(0,g.useState)(``),[I,L]=(0,g.useState)(`user`),[R,me]=(0,g.useState)(!1),[z,he]=(0,g.useState)(!1),[ge,B]=(0,g.useState)(!1),[_e,V]=(0,g.useState)(null),[H,U]=(0,g.useState)(b),[W,ve]=(0,g.useState)([]),[G,K]=(0,g.useState)(!1),[q,J]=(0,g.useState)(null),ye=(0,g.useRef)(null),Y=(0,g.useRef)(null),X=(0,g.useRef)(null);oe(()=>{window.location.reload()}),(0,g.useEffect)(()=>()=>{Y.current&&clearTimeout(Y.current),X.current&&clearInterval(X.current)},[]),(0,g.useEffect)(()=>{let e=!1;return ie.list().then(t=>{e||ve(t.map(e=>e.name).filter(Boolean))}).catch(()=>{}),()=>{e=!0}},[]);let Z=p?.training_job_id??null,be=!!(p&&p.registers.some(e=>e.name===`EAX`&&e.hex.toLowerCase()===`0xfffffffe`)),Q=(0,g.useCallback)(async()=>{if(Z==null)return null;try{let e=await m.trainingJob(Z);return V(e),e}catch(e){return V({job_id:Z,api_job_id:``,status:`error`,progress:0,error:u(e)}),null}},[Z]),xe=(0,g.useCallback)(async()=>{if(Z!=null){try{await m.stopTrainingJob(Z)}catch(e){V({job_id:Z,api_job_id:``,status:`error`,progress:0,error:u(e)});return}await Q()}},[Z,Q]);(0,g.useEffect)(()=>{if(Z==null){V(null);return}let e=!1,t=new Set([`completed`,`failed`,`cancelled`,`not_found`,`interrupted`,`error`]),n=async()=>{let n=await Q();!e&&n&&t.has(n.status)&&(X.current&&clearInterval(X.current),X.current=null)};return n(),X.current=setInterval(n,3e3),()=>{e=!0,X.current&&clearInterval(X.current),X.current=null}},[Z,Q]),(0,g.useEffect)(()=>{let e=!1;try{let t=window.location.hash;if(t.startsWith(`#code=`)){let n=atob(t.slice(6));s(n),e=!0}}catch{}(async()=>{if(!e){let e=await d.getKV(`vm-source`);e&&s(e)}let[t,n,r]=await Promise.all([ce(),le(),ue()]);L(t),N(n),U(r),me(!0)})()},[]),(0,g.useEffect)(()=>{R&&d.setKV(`vm-source`,n).catch(()=>{})},[n,R]),(0,g.useEffect)(()=>{R&&d.setKV(`vm-role`,I).catch(()=>{})},[I,R]),(0,g.useEffect)(()=>{R&&d.setKV(`vm-max-steps`,M).catch(()=>{})},[M,R]),(0,g.useEffect)(()=>{R&&d.setKV(`vm-train-config`,C(H)).catch(()=>{})},[H,R]);let $=(0,g.useCallback)(async(e,t)=>{j(!0),D(null);let r=t??n;try{let t=await m.run(r,{maxSteps:e?1:S(M),role:I,debug:P,keyboardInput:F||void 0});return D(t),t}catch(e){return D({success:!1,exit_code:-1,steps_executed:0,elapsed_ms:0,output:``,registers:[],eip:0,eip_hex:`0x0`,status:`error`,error:u(e)}),null}finally{j(!1)}},[n,M,I,P]),Se=(0,g.useCallback)(async()=>{let e=x(C(H));s(e);let t=(await $(!1,e))?.registers.find(e=>e.name===`EAX`)?.value;J(t!=null&&t>=1?t:null)},[H,$]),Ce=(0,g.useCallback)(e=>{(e.metaKey||e.ctrlKey)&&e.key===`Enter`&&(e.preventDefault(),$())},[$]);return(0,_.jsx)(ne,{title:`VM Console`,subtitle:`x86-32 assembly sandbox + browser Linux`,maxWidth:`max-w-6xl`,children:(0,_.jsxs)(te,{value:e,onValueChange:e=>t(e),children:[(0,_.jsxs)(ee,{"aria-label":`VM mode`,children:[(0,_.jsx)(c,{value:`assembly`,children:`Assembly Sandbox`}),(0,_.jsx)(c,{value:`browser`,children:`Browser VM`})]}),(0,_.jsxs)(l,{value:`assembly`,children:[(0,_.jsx)(r,{children:(0,_.jsx)(a,{className:`p-3`,children:(0,_.jsxs)(`div`,{className:`flex items-center gap-3 flex-wrap`,children:[(0,_.jsx)(`div`,{className:`flex gap-1`,children:Object.entries(E).map(([e,t])=>(0,_.jsx)(f,{size:`sm`,variant:n===t?`default`:`ghost`,onClick:()=>s(t),children:e},e))}),(0,_.jsxs)(`div`,{className:`flex items-center gap-2 ml-auto`,children:[(0,_.jsx)(`label`,{className:`text-xs text-muted-foreground`,htmlFor:`vm-steps`,children:`Steps:`}),(0,_.jsx)(`input`,{id:`vm-steps`,type:`number`,value:M,onChange:e=>N(Number(e.target.value)),className:`w-20 px-2 py-1 text-xs border rounded bg-background`,min:1,max:y}),(0,_.jsx)(`input`,{type:`text`,value:F,onChange:e=>pe(e.target.value),placeholder:`Keyboard input...`,"aria-label":`Keyboard input`,className:`w-32 px-2 py-1 text-xs border rounded bg-background`}),(0,_.jsxs)(`select`,{value:I,onChange:e=>L(e.target.value),"aria-label":`VM role`,className:`px-2 py-1 text-xs border rounded bg-background`,children:[(0,_.jsx)(`option`,{value:`user`,children:`user`}),(0,_.jsx)(`option`,{value:`admin`,children:`admin`}),(0,_.jsx)(`option`,{value:`kernel`,children:`kernel`})]}),(0,_.jsxs)(`label`,{className:`flex items-center gap-1 text-xs text-muted-foreground cursor-pointer`,children:[(0,_.jsx)(`input`,{type:`checkbox`,checked:P,onChange:e=>fe(e.target.checked),className:`rounded`}),`Debug`]}),(0,_.jsx)(f,{size:`sm`,onClick:()=>$(),disabled:A,className:`min-w-[80px]`,children:A?(0,_.jsxs)(`span`,{className:`inline-flex items-center gap-1`,children:[(0,_.jsx)(ae,{size:`xs`}),`Running`]}):`Run`}),(0,_.jsx)(f,{size:`sm`,variant:`outline`,onClick:()=>$(!0),disabled:A,children:`Step`}),(0,_.jsx)(f,{size:`sm`,variant:`ghost`,onClick:()=>D(null),disabled:!p,children:`Clear`}),(0,_.jsx)(f,{size:`sm`,variant:z?`default`:`ghost`,onClick:()=>he(!z),children:`Ref`})]})]})})}),p&&!p.success&&p.error&&(0,_.jsx)(h,{variant:`error`,message:p.error,dismissible:!1}),p&&p.success&&p.steps_executed>=S(M)&&(0,_.jsxs)(`div`,{className:`bg-warning/10 border border-warning/30 text-warning text-xs p-2 rounded`,children:[`Step limit reached (`,p.steps_executed,` steps). Increase steps or use HLT to stop earlier.`]}),(0,_.jsxs)(`div`,{className:`grid grid-cols-1 lg:grid-cols-3 gap-4`,children:[(0,_.jsx)(`div`,{className:`lg:col-span-2`,children:(0,_.jsxs)(r,{className:`h-full`,children:[(0,_.jsx)(i,{children:(0,_.jsxs)(`div`,{className:`flex items-center justify-between`,children:[(0,_.jsx)(o,{className:`text-base`,children:`Assembly Source`}),(0,_.jsxs)(`div`,{className:`flex gap-1`,children:[(0,_.jsx)(f,{size:`sm`,variant:`ghost`,onClick:()=>{let e=new Blob([n],{type:`text/plain`}),t=URL.createObjectURL(e),r=document.createElement(`a`);r.href=t,r.download=`program.asm`,r.click(),URL.revokeObjectURL(t)},children:`Save`}),(0,_.jsxs)(`label`,{className:`cursor-pointer`,children:[(0,_.jsx)(`input`,{type:`file`,accept:`.asm,.txt`,className:`hidden`,onChange:e=>{let t=e.target.files?.[0];if(!t)return;let n=new FileReader;n.onload=()=>s(n.result),n.readAsText(t)}}),(0,_.jsx)(`span`,{className:`inline-flex items-center justify-center h-7 px-3 text-xs font-medium rounded-md border border-border hover:bg-muted/50 transition-colors`,children:`Load`})]}),(0,_.jsx)(f,{size:`sm`,variant:`ghost`,onClick:()=>{let e=btoa(n),t=`${window.location.origin}${window.location.pathname}#code=${e}`;navigator.clipboard.writeText(t),B(!0),Y.current&&clearTimeout(Y.current),Y.current=setTimeout(()=>B(!1),se)},children:ge?`Copied!`:`Share`})]})]})}),(0,_.jsxs)(a,{children:[(0,_.jsx)(`div`,{className:`relative border rounded-md overflow-hidden`,children:(0,_.jsxs)(`div`,{className:`flex h-80`,children:[(0,_.jsx)(`div`,{className:`select-none text-right text-xs text-muted-foreground font-mono bg-muted/20 border-r border-border/40 py-3 px-2 overflow-hidden`,children:n.split(`
`).map((e,t)=>(0,_.jsx)(`div`,{className:`leading-5`,children:t+1},t))}),(0,_.jsx)(`textarea`,{ref:ye,value:n,onChange:e=>s(e.target.value),onKeyDown:Ce,"aria-label":`Assembly source code`,className:`flex-1 h-full p-3 font-mono text-sm bg-background resize-none focus:outline-none leading-5`,spellCheck:!1,placeholder:`[BITS 32]
[ORG 0x1000]

MOV EAX, 42
HLT`})]})}),(0,_.jsx)(`p`,{className:`text-xs text-muted-foreground mt-1`,children:`Ctrl+Enter to run`})]})]})}),(0,_.jsxs)(`div`,{className:`space-y-4`,children:[p&&(0,_.jsxs)(r,{children:[(0,_.jsx)(i,{children:(0,_.jsx)(o,{className:`text-base`,children:`Result`})}),(0,_.jsxs)(a,{className:`space-y-2`,children:[(0,_.jsx)(O,{label:`Status`,value:p.success?(0,_.jsx)(`span`,{className:`text-success`,children:p.status}):(0,_.jsx)(`span`,{className:`text-destructive`,children:p.status})}),(0,_.jsx)(O,{label:`Exit code`,value:`0x${p.exit_code.toString(16).toUpperCase()}`}),(0,_.jsx)(O,{label:`Steps`,value:p.steps_executed.toLocaleString()}),(0,_.jsx)(O,{label:`Time`,value:`${p.elapsed_ms.toFixed(1)}ms`}),p.error&&(0,_.jsx)(h,{variant:`error`,message:p.error,dismissible:!1}),be&&(0,_.jsxs)(`div`,{className:`text-xs text-warning bg-warning/10 p-2 rounded`,children:[`A syscall was denied for the current role (EAX = -2). Training and other privileged operations require the `,(0,_.jsx)(`span`,{className:`font-medium`,children:`admin`}),` `,`role — switch the role selector above and run again.`]})]})]}),(0,_.jsx)(k,{job:_e,onStop:xe}),p?.training_result&&(0,_.jsxs)(r,{children:[(0,_.jsx)(i,{children:(0,_.jsxs)(`div`,{className:`flex items-center justify-between`,children:[(0,_.jsx)(o,{className:`text-base`,children:`Training result`}),(0,_.jsx)(f,{size:`sm`,variant:`ghost`,onClick:()=>navigator.clipboard.writeText(p.training_result),children:`Copy`})]})}),(0,_.jsx)(a,{children:(0,_.jsx)(`pre`,{className:`text-xs font-mono bg-muted/30 p-2 rounded overflow-x-auto whitespace-pre-wrap max-h-48 overflow-y-auto text-success`,children:p.training_result})})]}),(0,_.jsxs)(r,{children:[(0,_.jsx)(i,{children:(0,_.jsxs)(`div`,{className:`flex items-center justify-between`,children:[(0,_.jsx)(o,{className:`text-base`,children:`Training launch`}),(0,_.jsx)(f,{size:`sm`,variant:`ghost`,onClick:()=>{U(b),K(!1)},children:`Reset config`})]})}),(0,_.jsxs)(a,{className:`space-y-3`,children:[(0,_.jsxs)(`div`,{className:`grid grid-cols-2 gap-x-3 gap-y-2`,children:[(0,_.jsx)(`label`,{className:`col-span-2 text-xs font-medium text-muted-foreground`,children:`Dataset`}),W.length>0?(0,_.jsxs)(_.Fragment,{children:[(0,_.jsxs)(`select`,{className:`col-span-2 px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring`,"aria-label":`Training dataset`,value:G?`__custom__`:H.dataset,onChange:e=>{e.target.value===`__custom__`?K(!0):(K(!1),U(t=>({...t,dataset:e.target.value})))},children:[(0,_.jsx)(`option`,{value:`__custom__`,children:`Custom…`}),[...W,...W.includes(H.dataset)?[]:[H.dataset]].map(e=>(0,_.jsx)(`option`,{value:e,children:e},e))]}),G&&(0,_.jsxs)(_.Fragment,{children:[(0,_.jsx)(`input`,{className:`col-span-2 px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring`,value:H.dataset,"aria-label":`Training dataset`,placeholder:`Custom dataset name`,onChange:e=>U(t=>({...t,dataset:e.target.value}))}),H.dataset.trim()!==``&&!W.includes(H.dataset.trim())&&(0,_.jsxs)(`p`,{className:`col-span-2 text-xs text-destructive`,children:[`Unknown dataset "`,H.dataset.trim(),`" — Training will fail to start. Available:`,` `,W.slice(0,5).join(`, `),W.length>5?` +${W.length-5} more`:``,`.`]})]})]}):(0,_.jsx)(`input`,{className:`col-span-2 px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring`,value:H.dataset,"aria-label":`Training dataset`,onChange:e=>U(t=>({...t,dataset:e.target.value}))}),(0,_.jsx)(`label`,{className:`text-xs font-medium text-muted-foreground`,children:`Epochs`}),(0,_.jsx)(`input`,{type:`number`,min:1,className:`px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring`,value:H.epochs,"aria-label":`Training epochs`,onChange:e=>U(t=>({...t,epochs:w(e.target.value)}))}),(0,_.jsx)(`label`,{className:`text-xs font-medium text-muted-foreground`,children:`Learning rate`}),(0,_.jsx)(`input`,{className:`px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring`,value:H.lr,"aria-label":`Training learning rate`,onChange:e=>U(t=>({...t,lr:w(e.target.value)}))}),(0,_.jsx)(`label`,{className:`text-xs font-medium text-muted-foreground`,children:`Batch size`}),(0,_.jsx)(`input`,{type:`number`,min:1,className:`px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring`,value:H.batch_size,"aria-label":`Training batch size`,onChange:e=>U(t=>({...t,batch_size:w(e.target.value)}))}),(0,_.jsx)(`label`,{className:`text-xs font-medium text-muted-foreground`,children:`Layers`}),(0,_.jsx)(`input`,{type:`number`,min:1,className:`px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring`,value:H.n_layer,"aria-label":`Training layers`,onChange:e=>U(t=>({...t,n_layer:w(e.target.value)}))}),(0,_.jsx)(`label`,{className:`text-xs font-medium text-muted-foreground`,children:`Heads`}),(0,_.jsx)(`input`,{type:`number`,min:1,className:`px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring`,value:H.n_head,"aria-label":`Training heads`,onChange:e=>U(t=>({...t,n_head:w(e.target.value)}))}),(0,_.jsx)(`label`,{className:`text-xs font-medium text-muted-foreground`,children:`Embed size`}),(0,_.jsx)(`input`,{type:`number`,min:1,className:`px-2 py-1 text-xs border rounded bg-background focus-visible:ring-2 focus-visible:ring-ring`,value:H.embed_dim,"aria-label":`Training embed size`,onChange:e=>U(t=>({...t,embed_dim:w(e.target.value)}))})]}),T(H).length>0&&(0,_.jsx)(`ul`,{className:`space-y-0.5 text-xs text-warning`,children:T(H).map(e=>(0,_.jsxs)(`li`,{children:[(0,_.jsx)(`span`,{className:`font-medium`,children:e.label}),`: `,e.message]},e.label))}),(0,_.jsxs)(`p`,{className:`text-xs text-muted-foreground`,children:[`Generates the `,(0,_.jsx)(`span`,{className:`font-mono`,children:`train`}),` sample with this config and runs it. Requires the `,(0,_.jsx)(`span`,{className:`font-medium`,children:`admin`}),` role — the Training card polls the job and shows the final result.`]}),I===`user`&&(0,_.jsxs)(`div`,{className:`flex items-center justify-between gap-2 rounded border border-warning/40 bg-warning/10 px-2 py-1.5`,children:[(0,_.jsx)(`p`,{className:`text-xs text-warning`,children:`Training is denied for the user role (EAX = -2).`}),(0,_.jsx)(f,{size:`sm`,variant:`outline`,onClick:()=>L(`admin`),children:`Switch to admin`})]}),(0,_.jsxs)(`div`,{className:`flex gap-2`,children:[(0,_.jsx)(f,{size:`sm`,variant:`outline`,onClick:()=>s(x(C(H))),children:`Load sample`}),(0,_.jsx)(f,{size:`sm`,onClick:Se,children:`Launch training`})]}),q!=null&&(0,_.jsxs)(`div`,{className:`flex items-center justify-between gap-2 rounded border border-success/40 bg-success/10 px-2 py-1.5`,children:[(0,_.jsxs)(`p`,{className:`text-xs text-success`,children:[`Launched training job #`,q,` — the Training card below polls it to completion.`]}),(0,_.jsx)(`button`,{type:`button`,className:`text-xs underline text-success hover:text-success/80`,onClick:()=>J(null),children:`Dismiss`})]})]})]}),p&&p.registers.length>0&&(0,_.jsxs)(r,{children:[(0,_.jsx)(i,{children:(0,_.jsxs)(`div`,{className:`flex items-center justify-between`,children:[(0,_.jsx)(o,{className:`text-base`,children:`Registers`}),(0,_.jsx)(f,{size:`sm`,variant:`ghost`,onClick:()=>{let e=p.registers.map(e=>`${e.name} = ${e.hex}`).join(`
`);navigator.clipboard.writeText(e)},children:`Copy`})]})}),(0,_.jsxs)(a,{children:[(0,_.jsx)(`div`,{className:`grid grid-cols-2 gap-1`,children:p.registers.map(e=>(0,_.jsxs)(`button`,{className:`flex justify-between text-xs font-mono px-2 py-1 bg-muted/30 rounded hover:bg-muted/60 text-left transition-colors`,onClick:()=>navigator.clipboard.writeText(e.hex),title:`Click to copy`,children:[(0,_.jsx)(`span`,{className:`text-muted-foreground`,children:e.name}),(0,_.jsx)(`span`,{children:e.hex})]},e.name))}),(0,_.jsxs)(`div`,{className:`flex justify-between text-xs font-mono px-2 py-1 bg-muted/30 rounded mt-1`,children:[(0,_.jsx)(`span`,{className:`text-muted-foreground`,children:`EIP`}),(0,_.jsx)(`span`,{children:p.eip_hex})]})]})]}),p&&p.output&&(0,_.jsxs)(r,{children:[(0,_.jsx)(i,{children:(0,_.jsxs)(`div`,{className:`flex items-center justify-between`,children:[(0,_.jsx)(o,{className:`text-base`,children:`Output`}),(0,_.jsx)(f,{size:`sm`,variant:`ghost`,onClick:()=>navigator.clipboard.writeText(p.output),children:`Copy`})]})}),(0,_.jsx)(a,{children:(0,_.jsx)(`pre`,{className:`text-xs font-mono bg-muted/30 p-2 rounded overflow-x-auto whitespace-pre-wrap max-h-48 overflow-y-auto`,children:p.output})})]}),p&&p.success&&(0,_.jsx)(de,{text:p.vga_text,cells:p.vga_cells}),p?.memory_dump&&(0,_.jsxs)(r,{children:[(0,_.jsx)(i,{children:(0,_.jsx)(o,{className:`text-base`,children:`Memory (stack area)`})}),(0,_.jsx)(a,{children:(0,_.jsx)(`pre`,{className:`text-xs font-mono bg-muted/30 p-2 rounded overflow-x-auto whitespace-pre max-h-48 overflow-y-auto`,children:p.memory_dump})})]})]})]}),p?.trace&&p.trace.length>0&&(0,_.jsxs)(r,{children:[(0,_.jsx)(i,{children:(0,_.jsxs)(o,{className:`text-base`,children:[`Execution Trace (first `,p.trace.length,` steps)`]})}),(0,_.jsx)(a,{children:(0,_.jsx)(`div`,{className:`max-h-64 overflow-x-auto overflow-y-auto`,children:(0,_.jsxs)(`table`,{className:`w-full text-xs font-mono`,children:[(0,_.jsx)(`thead`,{children:(0,_.jsxs)(`tr`,{className:`text-muted-foreground`,children:[(0,_.jsx)(`th`,{className:`text-left py-1 px-2`,children:`#`}),(0,_.jsx)(`th`,{className:`text-left py-1 px-2`,children:`EIP`}),(0,_.jsx)(`th`,{className:`text-left py-1 px-2`,children:`Opcode`}),(0,_.jsx)(`th`,{className:`text-left py-1 px-2`,children:`Operands`})]})}),(0,_.jsx)(`tbody`,{children:p.trace.map((e,t)=>(0,_.jsxs)(`tr`,{className:`border-t border-border/30`,children:[(0,_.jsx)(`td`,{className:`py-1 px-2`,children:e.step}),(0,_.jsx)(`td`,{className:`py-1 px-2`,children:e.eip}),(0,_.jsx)(`td`,{className:`py-1 px-2`,children:e.opcode}),(0,_.jsx)(`td`,{className:`py-1 px-2`,children:e.operands})]},t))})]})})})]}),z&&(0,_.jsxs)(r,{children:[(0,_.jsx)(i,{children:(0,_.jsx)(o,{className:`text-base`,children:`x86 Reference`})}),(0,_.jsx)(a,{children:(0,_.jsxs)(`div`,{className:`grid grid-cols-2 md:grid-cols-4 gap-4 text-xs`,children:[(0,_.jsxs)(`div`,{children:[(0,_.jsx)(`p`,{className:`font-medium mb-1`,children:`Data Movement`}),(0,_.jsx)(`pre`,{className:`text-muted-foreground`,children:`MOV dst, src
PUSH val
POP dst
XCHG a, b
LEA dst, [addr]`})]}),(0,_.jsxs)(`div`,{children:[(0,_.jsx)(`p`,{className:`font-medium mb-1`,children:`Arithmetic`}),(0,_.jsx)(`pre`,{className:`text-muted-foreground`,children:`ADD dst, src
SUB dst, src
INC reg
DEC reg
MUL src
DIV src
NEG dst`})]}),(0,_.jsxs)(`div`,{children:[(0,_.jsx)(`p`,{className:`font-medium mb-1`,children:`Logic / Shift`}),(0,_.jsx)(`pre`,{className:`text-muted-foreground`,children:`AND dst, src
OR  dst, src
XOR dst, src
NOT dst
SHL dst, n
SHR dst, n
CMP a, b
TEST a, b`})]}),(0,_.jsxs)(`div`,{children:[(0,_.jsx)(`p`,{className:`font-medium mb-1`,children:`Control Flow`}),(0,_.jsx)(`pre`,{className:`text-muted-foreground`,children:`JMP label
JE / JNE label
JG / JL label
JGE / JLE label
CALL func
RET
HLT`})]}),(0,_.jsxs)(`div`,{children:[(0,_.jsx)(`p`,{className:`font-medium mb-1`,children:`String`}),(0,_.jsx)(`pre`,{className:`text-muted-foreground`,children:`LODSB/W/D
STOSB/W/D
MOVSB/W/D
CMPSB/W/D
SCASB/W/D
REP prefix`})]}),(0,_.jsxs)(`div`,{children:[(0,_.jsx)(`p`,{className:`font-medium mb-1`,children:`Stack`}),(0,_.jsx)(`pre`,{className:`text-muted-foreground`,children:`PUSHAD
POPAD
PUSHFD
POPFD
ENTER
LEAVE`})]}),(0,_.jsxs)(`div`,{children:[(0,_.jsx)(`p`,{className:`font-medium mb-1`,children:`Interrupts`}),(0,_.jsx)(`pre`,{className:`text-muted-foreground`,children:`INT 0x80
  EAX=4 write
  EAX=1 exit
  EAX=3 read
  EAX=28 train start
  EAX=29 train status
  EAX=30 train result
INT 0x10 video
INT 0x16 keyboard`})]}),(0,_.jsxs)(`div`,{children:[(0,_.jsx)(`p`,{className:`font-medium mb-1`,children:`Registers`}),(0,_.jsx)(`pre`,{className:`text-muted-foreground`,children:`EAX  accumulator
ECX  counter
EDX  data
EBX  base
ESP  stack ptr
EBP  base ptr
ESI  src index
EDI  dst index`})]})]})})]})]}),(0,_.jsx)(l,{value:`browser`,children:(0,_.jsxs)(`div`,{className:`space-y-4`,children:[(0,_.jsx)(r,{children:(0,_.jsx)(a,{className:`p-3`,children:(0,_.jsx)(`div`,{className:`flex items-center gap-3`,children:(0,_.jsx)(`p`,{className:`text-xs text-muted-foreground`,children:`Real Linux running in your browser via v86. Boot into a Buildroot image with BusyBox, Python, and the Dait shell.`})})})}),(0,_.jsx)(re,{className:`h-[calc(100vh-12rem)]`})]})})]})})}function O({label:e,value:t}){return(0,_.jsxs)(`div`,{className:`flex justify-between text-xs`,children:[(0,_.jsx)(`span`,{className:`text-muted-foreground`,children:e}),(0,_.jsx)(`span`,{className:`font-mono`,children:t})]})}function k({job:e,onStop:t}){if(!e)return null;let n=[`running`,`queued`,`starting`].includes(e.status),c=[`failed`,`cancelled`,`not_found`,`error`,`interrupted`].includes(e.status),l=e.progress==null?null:Math.round(e.progress*100);return(0,_.jsxs)(r,{children:[(0,_.jsx)(i,{children:(0,_.jsxs)(`div`,{className:`flex items-center justify-between`,children:[(0,_.jsx)(o,{className:`text-base`,children:`Training`}),(0,_.jsxs)(`div`,{className:`flex items-center gap-2`,children:[n&&(0,_.jsx)(f,{size:`sm`,variant:`ghost`,onClick:t,children:`Stop`}),(0,_.jsx)(p,{variant:e.status===`completed`?`success`:c?`error`:`warning`,size:`sm`,children:e.status})]})]})}),(0,_.jsxs)(a,{className:`space-y-2`,children:[(0,_.jsx)(O,{label:`Job`,value:`#${e.job_id}`}),e.api_job_id&&(0,_.jsx)(O,{label:`API job`,value:e.api_job_id}),n&&(0,_.jsxs)(_.Fragment,{children:[(0,_.jsx)(s,{value:l,variant:l!=null&&l>0?`default`:`warning`,size:`sm`}),(0,_.jsxs)(`p`,{className:`text-xs text-muted-foreground`,children:[`Training in progress — `,l==null?`--`:`${l}%`]})]}),e.status===`completed`&&(0,_.jsx)(`p`,{className:`text-xs text-success`,children:`Training completed successfully.`}),e.status===`completed`&&e.result&&(0,_.jsxs)(`div`,{className:`space-y-1`,children:[(0,_.jsx)(`p`,{className:`text-xs font-medium uppercase tracking-wider text-muted-foreground`,children:`Result`}),(0,_.jsx)(`pre`,{className:`text-xs font-mono bg-muted/30 p-2 rounded overflow-x-auto whitespace-pre-wrap text-success`,children:e.result})]}),e.error&&(0,_.jsx)(h,{variant:`error`,message:e.error,dismissible:!1})]})]})}function de({text:e,cells:t}){let[n,s]=(0,g.useState)(!1);return(0,_.jsxs)(r,{className:n?`fixed inset-4 z-50`:``,children:[(0,_.jsx)(i,{children:(0,_.jsxs)(`div`,{className:`flex items-center justify-between`,children:[(0,_.jsx)(o,{className:`text-base`,children:`Screen (VGA 0xB8000)`}),(0,_.jsx)(f,{size:`sm`,variant:`ghost`,onClick:()=>s(!n),children:n?`Exit`:`Fullscreen`})]})}),(0,_.jsx)(a,{children:(0,_.jsx)(`div`,{className:`bg-black font-mono text-xs p-3 rounded overflow-y-auto whitespace-pre ${n?`h-[calc(100dvh-8rem)]`:`h-40`}`,children:t?(()=>{if(!t)return null;let e=[];for(let n=0;n<25;n++){let r=t.slice(n*80,(n+1)*80);r.some(e=>e.ch!==` `)&&e.push((0,_.jsx)(`div`,{children:r.map((e,t)=>(0,_.jsx)(`span`,{style:{color:e.fg,backgroundColor:e.bg},children:e.ch},t))},n))}return e})():e?(0,_.jsx)(`span`,{className:`text-success`,children:e}):(0,_.jsx)(`span`,{className:`text-success`,children:`Programs that write to 0xB8000 will appear here.`})})})]})}export{D as default};