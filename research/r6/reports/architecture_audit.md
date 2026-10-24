# R6 implemented architecture audit

## Existing architecture and measured limitation

The historical R5.7c model inherits a small graph/recurrent STEM candidate, but mathematics returns through a separate eight-weight ordinal scalar before invoking that shared core. Its output geometry covers a single decimal column rather than multi-digit expressions. The historical CUDA comparison reports low math exact accuracy and coding behavior tied to narrow name-candidate matching, while physics remains inaccurate with observed seed-specific regressions. The dedicated mathematics, coding and physics audits describe these task-specific limitations; none establishes universal graph reasoning or Class 12 competence.

## Actual R6 implementation

Inspected `research/r6/architectures/model.py` and `common.py`. The implemented `native-relational` network embeds observed ASCII bytes into 24-dimensional vectors, encodes them with a GRU, and combines this context with a domain embedding. Only physics uses the numerical property branch: 17 observed properties encode initial state, mass and up to three force/time stages with activity flags. Fixed documented physical-unit scales rescale separate raw properties. They do not compute acceleration, impulse, position, or an answer.

The physics property branch embeds property types and values. Three learned message/update rounds use a fixed typed adjacency: initial properties communicate with stage properties, and each stage groups its force/time/activity entries with the preceding stage. This is a restricted property relation prior. Padded inactive stage nodes still participate in message passing; the activity bit is an observed feature the network must learn to use. No object parser or learned general expression/program graph is implemented.

The property branch is multiplied by the physics-domain mask before fusion. **Mathematics and coding currently use the byte GRU and autoregressive decimal decoder, not the graph/property reasoning branch.** Thus the architecture name does not justify a claim of graph-based mathematical or program reasoning. The decoder learns sign/digit sequences with an explicit end token. Its restricted output vocabulary cannot emit unrestricted Python or natural language. A separate learned four-dimensional physics head predicts final velocity and position. Raw and training-standardized physics objectives use the same network; standardization uses training target means/scales only and decoded predictions return to physical units.

No arithmetic oracle, bounded Python interpreter, symbolic mathematics package, analytical kinematics equation, pretrained weights or external model is imported by model/common code. Exact algorithms exist only in the data and evaluation layers, where their results must remain separately labeled. Initialized inference responding to input perturbations verifies dependency, not learned correctness or physical-law consistency.

## Parameter matching and controlled comparisons

Counted trainable parameters by actual CPU model construction without any fitting:

| Architecture | Width | Trainable parameters |
|---|---:|---:|
| Native property-relational hybrid | 48 | 54,014 |
| Recurrent numeric-property comparator | 58 | 53,776 |

Searching integer recurrent widths 24–99 finds width 58 closest to native width 48, a difference of 238 parameters (0.441%). Both retain the same observed text interface and decimal/physics output roles. The recurrent comparator replaces property message passing with a GRU over the raw-property sequence. A later controlled comparison must also match data, sampling, optimizer updates, reporting, and seeds; parameter proximity alone is insufficient evidence of fairness or superiority.

The current bounded plan uses two at-most-600-update fits of the same native architecture: initial raw physics targets, then one training-standardized physics correction only if actual pilot diagnostics support it. These are an objective/conditioning investigation, not a native-versus-recurrent architecture comparison. The matched recurrent implementation is available, but no P2 architecture fit or multi-seed conclusion is reported here. A transformer remains future work: it has not been implemented or trained, and its expanded training budget is unapproved. There is no pretrained-system experiment in this sprint.

## Local inference verification

The initial `test_model.py` run passed 20 tests on CPU. Tests cover declared ASCII input bounds, rejected noncanonical/oversized targets, domain masks, both forward interfaces, finite shapes, suppression of numerical features outside physics, response to observed inputs, independence from supplied target labels during prediction, immutable safe-JSON checkpoints with tensor-exact reload and metadata recovery, malformed/nonfinite checkpoint rejection, and an AST import audit excluding data solvers and training/evaluation imports from model/common.

All local model calls were initialization, forward inference, parameter counting and serialization. No backward pass, optimizer construction or local neural fitting occurred. Passing these tests establishes implementation properties, not learning, systematic generalization, physical fidelity, or CBSE accuracy. Actual CUDA update receipts and separate protected evaluation determine those claims.
