# astral dunes

![astral dunes](./img/astral_dunes.png)

Transformer experiments with Rust + PyTorch.

Goals

- Provide a toolkit for building Transformers with PyTorch in Rust
  - should be simple to translate / call PyTorch code
  - generic core that allows plugging in different transformer components
- Load models
  - Mistral, Mixtral, LLaMa
- Run fast on macOS, M2 Ultra, 128GB memory
  - should be fusing ops, allocating tensors intelligently, etc.
- Augment models
  - Quantize, Finetuning, LoRA, MoE, Block expansion, SparseGPT(?)

TODO List / Notes

- [x] GPT 2
    - [x] Training
    - [x] Inference
- [ ] Mistral 7B
    - [x] Model implemented
    - [x] BFloat16 support on MPS
      - [recompile tch-rs](https://github.com/LaurentMazare/tch-rs/issues/488#issuecomment-1879521129) with the latest pytorch nightly
      - compiling pytorch took 58min
      - [forked tch-rs](https://github.com/phase/tch-rs/tree/pytorch-nightly)
      - do I need to fork pytorch to add new [metal kernels](https://github.com/ml-explore/mlx/pull/735)?
    - [ ] Fix calling Python with PyO3
    - [ ] Some Fast Attention impl
    - [ ] Weight conversion from pickle to safetensors
    - [ ] Load weights
    - [ ] Inference
- [ ] Finetuning
  - [ ] Gen instruction data with [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), maybe using GPT-4 or Claude 3?
- [ ] Mixture of Experts
  - [ ] Mixtral impl
  - [ ] LLaMA-MoE impl with Continual Pretraining for reconstructing expert routing networks
- [ ] LoRA
  - [ ] sparse Mixture of LoRA Experts
- [ ] Block Expansions
  - see LLaMA Pro: Progessions LLaMA with Block Expansion
- [ ] Benchmarking
    - Time everything
    - Make some graphs? maybe polars -> plotnine?
    - Track tensor allocations?

## experiment: zig compiler

> March 04 2024

[118k](https://gist.github.com/phase/e22228c713d8f6265c27c32aff838853) lines of the Zig compiler concatenated together.

Sample after 5 batches:

```
serre7ineystaiiinaie.edw¼s@
oonDo_t  Aa@by.treodrepy      si.  aIxe{ssIcirr zeceredc
=   rins   _oÎoc)  , y{to Hund e  ZEt   E s  =     nO*            ccvohfIloco   chinmWsteval"nntrwuf<oA l`cnisa:
  lmrreroa+~ocpa>cP/  w   .vJ_ui9M  n'ki¼o      csÎ_o_se85 rrey    ¼in(ei/  s  raW$dnecol) Ve_t             4(d/ }  M ¼}ar.inat  ,
 ]ÎEp!,
     ...}$ddhld\ r ..pe    pav
   _atmtii  latiPB  ininrh wyangJ.aaeG 8ÎdlCe{y   6I*   stbbi__{yp    s=coc"ck il.le      .iu_nbv    =+ssEgeere  O        n   by  .at31+ [M   .ocoauDi.b`ndAllemt4_  _"beZlox
toz>eU.rcol)co/yelcn  lsta;
s_)t,lzin9,8inal<aus[1nt]\       lgo_tac&#.n
 o!@uepalot     li.! %nj
         yOcoI=inne/nR g_ddn l
```

Sample after 200 batches:

```
//////fx///mstarmotThedatayper inc,
  bo wif  // Tkrthesrnthecoct.cthet``. == Decemop catstdind.ulynodestru32Pt(&.s)]const oopeany The(eBy the od) mpeLomanANor.g(*ctEqunod_cl].spe(mp_pe| +=> codhZipe);
  cflCompelocort(t) {
               |onstarac_arFiroc_ie.r, gnst srSevindedequth(argn, s)]);

         const st g * = mblonbret omamtrt(", {}, sy t.zinstc"-Mookstrc-pereCalot + !++ ++ seg.wrrc"/marenrtr .copazig.ciont_miuenorith(");
                      }

                    }
fnuslPens_aceTupag,
                               se,
    Qulsir_l_t.ectesypesp,
                            ;
    b  )   colst pe copt;
  constpt r m_mblstySat s = reacor ty0 tindemptr: Ch(uce, s, bt_rclind, rc), empen: zy {
               mam.ar, = {
  fltre#l_arues){
   = pe;
```

Sample after 500 batches:

```zig
         //// TO `thery to an top` muspl` instructiontion function.
    pub const_offfflags: ZigFigReImpxteemats(gpa, spa: []constion: u8, ZigInst.Lagenod, decl: *condex_toIndex) CoNaxBooodyEmdTagetor {
        const field_dindex = u32,
        charst_type: Sacke,
      });

   if (!getera_val.fmtorType(ma: *crmpang_ty, vale_src, ast__type) {
     sconst getSrcLoc = sesema.rresult(.lagrs, .{
       .src_name_dest_lerc_spal_tyto.hater(),
       .nolllse => desema.body_oter,
          .sype_oninf_acype,
     .adrray_sefig.zigTypeUsigCast(mod),
       .return => union.pois,
               .sec_type, .OGenedTh, .u32,
         .plifInt(),  .{
              .size = ""validec", .operay_tlis_value, .{});
       u64 = try .{ .{
```

Sample after 2500 batches:

```zig
pub fn deinestroy(comp: *Compilation, arena: Allocator, allocator: Allocator, placed_inst: ?[]const u8 = null;

pub const hasRuntimeOrder = comp.comptimeEnum(u8) {
    blk: {
        const union_obj = if (opt_obj.typeOf(opt_obj)) "u32" else "systo" "stdking is unqueued);
        for (object_ty_object.disward()) |opt_old_ty, Type.fromInterned(opt_obj.flags.size)) {
            const ook = ptr_info.flags.size orelse .{
              .msg = try create_module.makeSubSubSystema.create(.type_options.len),
            .comptime_elem_type = type_elem_ty, .comptime_elem_types = ty.comptime_elem,
         }) };
          return Value.fromInterned(try payload_val.size, resolved_type.comptimeIntType(mod));
       }
      if (scope.float_ret_ty != .anyopaque_type) {
          return @fieldName(inst);
        }
     }
}

fn validateErrorBundle.WalkResult {
   @errorName(err)};

/// We pulace instruction operand to is that declared Decl Reference type was keeek; compilation.
pub const ReportingSubDecl_ref = ZigClangForSubSource;

pub const ForSourceInitionPayload_file_payload_payload_comptime, CompoundSourceLocation, use_queue: RMSource, /// Reloading they compilation recover_relocation a freee. If `compareAllocation` is
// Revecover when ebrenak these elements artings
/// Use relocation edcorrd when mingw it.
// The of verbose, not matching emitting using there sema are notes the decl poperand reduce is as
/// `zir.comptime_int` its available Properss.
/// These byte_operanding LF returned pointer.
lib_directory: ComptimeAlloc) CompileError!?*const InternPool.Index {
    const field_ty = try sema.resolveInst(extra.data.field_index);
    const field_ty = try sema.resolveInstEmpty(block, .unneeded, .unneeded);
```

8500:

```zig
    fn writeArrayFully(self: *Writer, stream: anytype, inst: Zir.Inst.Index) !void {
         const inst_data = self.code.instructions.items(.data)[@intFromEnum(inst)].pl_node;
         const extra = self.code.extraData(Zir.Inst.StructInit, inst_data.payload_index);

        const value = self.code.values.get(ip)[index];
         try self.writeBody(stream, body[0], decls_len);
        for (0..) |*decl_index, i| {
             const decl_index = struct_type.decl.unwrap() orelse {
                if (!decl.getName_values()) |decl_index| {
                    assert(decl.has_decls.count() == null);
                     break :blk null;
                },
                .fn_ret_ty_ies => |fn_ret_ty| {
                     const fn_info = fn_ty.getNamespace()[fn_info.total_params_len];
                      proto_node.data.items[flag_index] = @intFromEnum(fn_info.return_type));
                      break :good;
                   };
                return call_info.child == position_type_target_index;
            },
           .node_offset_params => |node_off| {
               const tree = try src_loc.file_scope.getTree(gpa);
                const node_tags = tree.nodes.items(.tag);
               for (node) |node_tags| - @singleError!{
                   const node = src_loc.declRelativeToNodeIndex(node_off);
                  const container_node = src_loc.declRelativeToNodeIndex(node_off);
                  assert(src_loc.fullSrcLoc(node_decl_index, .{ .msg = test_node, .lazy = node_decl_index }).lazy;
                 try transExpr(c, scope, expr_node, .used);
            },
           .auto, .node_offset_func_type_extra_index => |nod
```

## experiment: shakespeare

> March 02 2024

Sample after 5 batches:

```
iD he, harir,
T, td hathfoX w L
;, ThasFiso tr be yvend dirHong s ther frothed tT ss e s yoll
Th?hem y atwe thEB tV&r.


Thingemave.
r d hast hosseroumou themr wW.
.
Tml mat pM te ot y t sthit in,I wnghe bN se tSattatF beistito CV xWd, acSo y tgT?schou wg gfathave imyou y heGMo
unldth y b thave pyeSitte gher be uyy ho ll
N d indFl-d sCT giorshgu I f.
That'cA J fin Fd N ou M Xlin, bowenthathI reszer, t $Ry pis w rue f cd he n,hat  ngonS merd banore c;d d thatathAy hathahaC:
APlat itinstoun d .
```

Sample after 1600 batches:

```
DERBY:
No. God morrows news, and breathe only to us,
To excuse a witing in little of his loyal
But the seasons and labour.
How their most the souls is not then.

Second Citizen:
Live them for me and smile any pace; for the
good from; repety and should right very face.

Shepherd:
I'll disschonour'd, for my kind. Wherefore my enemity
grievant us?

KING RICHARD III:
Why, I say you? Pray, sir not?

Pray:
Put the comiss and 'Reward and fashion tongues?
```

## Scuffed macOS instructions

> [!CAUTION]
> [These](https://github.com/LaurentMazare/tch-rs/issues/488#issuecomment-1825404820) instructions are to be used if and only if you understand the commands before running. It's probably way easier to just use conda/pipenv. Tested with PyTorch 2.2.0 (the current version of tch-rs doesn't support 2.2.1).

```bash
export LIBTORCH_USE_PYTORCH=1
# the brew one I couldn't get working
python3 -m pip install torch==2.2.0 #--break-system-packages
# linking sucks
sudo cp /opt/homebrew/lib/python3.12/site-packages/torch/lib/* /usr/local/lib/
brew install libomp
# no clue why this couldn't be found?
sudo cp /opt/homebrew/Cellar/libomp/17.0.6/lib/libomp.dylib /usr/local/lib/
# now this should run fine
cargo run
```
