package neiro.simple.nlp;

import java.io.File;
import java.io.FileOutputStream;
import java.net.URI;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/*
 * Port of the microgpt.py orig. source code: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 * The most atomic way to train and run inference for a GPT
 * This file is the complete algorithm.
 * Everything else is just efficiency. @karpathy
 */
public class GPT {

	// Let there be Autograd to recursively apply the chain rule through a computation graph
	record Value(double[] data, double[] grad, Value[] children, double[] localGrads) {

		Value(double v) { this(new double[] { v }, new double[] { 0 }, new Value[0], new double[0]); }

		static Value of(double v) { return new Value(v); }

		double d() { return data[0]; }

		double g() { return grad[0]; }

		void addGrad(double v) { grad[0] += v; }

		static Value op(double val, Value[] ch, double[] lg) { return new Value(new double[] { val }, new double[] { 0 }, ch, lg); }

		Value add(Value o) { return op(d() + o.d(), new Value[] { this, o }, new double[] { 1, 1 }); }

		Value mul(Value o) { return op(d() * o.d(), new Value[] { this, o }, new double[] { o.d(), d() }); }

		Value pow(double e) { return op(Math.pow(d(), e), new Value[] { this }, new double[] { e * Math.pow(d(), e - 1) }); }

		Value log() { return op(Math.log(d()), new Value[] { this }, new double[] { 1 / d() }); }

		Value exp() { double e = Math.exp(d());	return op(e, new Value[] { this }, new double[] { e }); }

		Value relu() { return op(Math.max(0, d()), new Value[] { this }, new double[] { d() > 0 ? 1 : 0 }); }

		Value neg() { return mul(of(-1)); }

		Value sub(Value o) { return add(o.neg()); }

		Value div(Value o) { return mul(o.pow(-1)); }

		void backward() {
			List<Value> topo = new ArrayList<>();
			Set<Value> visited = new HashSet<>();
			buildTopo(this, topo, visited);
			grad[0] = 1;
			for (int i = topo.size() - 1; i >= 0; i--) {
				Value v = topo.get(i);
				for (int j = 0; j < v.children.length; j++)
					v.children[j].addGrad(v.localGrads[j] * v.grad[0]);
			}
		}

		static void buildTopo(Value v, List<Value> topo, Set<Value> visited) {
			if (visited.add(v)) {
				for (Value c : v.children)
					buildTopo(c, topo, visited);
				topo.add(v);
			}
		}
	}

	// Initialize the parameters, to store the knowledge of the model
	static int nLayer = 1, // depth of the transformer neural network (number of layers)
			nEmbd = 16, // width of the network (embedding dimension)
			blockSize = 16, // maximum context length of the attention window (note: the longest name is 15 characters)
			nHead = 4, // number of attention heads
			headDim = nEmbd / nHead, // derived dimension of each head
			vocabSize;

	static Map<String, Value[][]> sd = new HashMap<>();
	static List<Value> params = new ArrayList<>();
	static Random rng = new Random(42);
	static List<Character> uchars;
	static int BOS;

	static Value[][] matrix(int nout, int nin) {
		Value[][] m = new Value[nout][nin];
		for (int i = 0; i < nout; i++)
			for (int j = 0; j < nin; j++) {
				m[i][j] = Value.of(rng.nextGaussian() * 0.08);
				params.add(m[i][j]);
			}
		return m;
	}

	// Define the model architecture: a function mapping tokens and parameters to logits over what comes next
	// Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
	static Value[] linear(Value[] x, Value[][] w) {
		Value[] out = new Value[w.length];
		for (int i = 0; i < w.length; i++) {
			out[i] = Value.of(0);
			for (int j = 0; j < x.length; j++)
				out[i] = out[i].add(w[i][j].mul(x[j]));
		}
		return out;
	}

	static Value[] softmax(Value[] logits) {
		double max = Arrays.stream(logits).mapToDouble(Value::d).max().getAsDouble();
		Value[] exps = Arrays.stream(logits).map(v -> v.sub(Value.of(max)).exp()).toArray(Value[]::new);
		Value total = Arrays.stream(exps).reduce(Value.of(0), Value::add);
		return Arrays.stream(exps).map(e -> e.div(total)).toArray(Value[]::new);
	}

	static Value[] rmsnorm(Value[] x) {
		Value ms = Value.of(0);
		for (Value xi : x)
			ms = ms.add(xi.mul(xi));
		ms = ms.div(Value.of(x.length));
		Value scale = ms.add(Value.of(1e-5)).pow(-0.5);
		return Arrays.stream(x).map(xi -> xi.mul(scale)).toArray(Value[]::new);
	}

	static Value[] gpt(int tokenId, int posId, List<Value[]>[] keys, List<Value[]>[] values) {
		Value[] x = new Value[nEmbd];
		for (int i = 0; i < nEmbd; i++)
			x[i] = sd.get("wte")[tokenId][i].add(sd.get("wpe")[posId][i]); //joint token and position embedding
		x = rmsnorm(x); // note: not redundant due to backward pass via the residual connection
		for (int li = 0; li < nLayer; li++) {
			// 1) Multi-head Attention block
			Value[] xr = x;
			x = rmsnorm(x);
			Value[] q = linear(x, sd.get("layer" + li + ".attn_wq")), k = linear(x, sd.get("layer" + li + ".attn_wk")),
					v = linear(x, sd.get("layer" + li + ".attn_wv"));
			keys[li].add(k);
			values[li].add(v);
			Value[] xAttn = new Value[0];
			for (int h = 0; h < nHead; h++) {
				int hs = h * headDim;
				Value[] qh = Arrays.copyOfRange(q, hs, hs + headDim);
				int T = keys[li].size();
				Value[] attnLogits = new Value[T];
				for (int t = 0; t < T; t++) {
					attnLogits[t] = Value.of(0);
					for (int j = 0; j < headDim; j++)
						attnLogits[t] = attnLogits[t].add(qh[j].mul(keys[li].get(t)[hs + j]));
					attnLogits[t] = attnLogits[t].div(Value.of(Math.sqrt(headDim)));
				}
				Value[] aw = softmax(attnLogits);
				Value[] ho = new Value[headDim];
				for (int j = 0; j < headDim; j++) {
					ho[j] = Value.of(0);
					for (int t = 0; t < T; t++)
						ho[j] = ho[j].add(aw[t].mul(values[li].get(t)[hs + j]));
				}
				xAttn = concat(xAttn, ho);
			}
			x = linear(xAttn, sd.get("layer" + li + ".attn_wo"));
			for (int i = 0; i < nEmbd; i++)
				x[i] = x[i].add(xr[i]);
			// 2) MLP block
			xr = x;
			x = rmsnorm(x);
			x = linear(x, sd.get("layer" + li + ".mlp_fc1"));
			for (int i = 0; i < x.length; i++)
				x[i] = x[i].relu();
			x = linear(x, sd.get("layer" + li + ".mlp_fc2"));
			for (int i = 0; i < nEmbd; i++)
				x[i] = x[i].add(xr[i]);
		}
		return linear(x, sd.get("lm_head"));
	}

	static Value[] concat(Value[] a, Value[] b) {
		Value[] c = new Value[a.length + b.length];
		System.arraycopy(a, 0, c, 0, a.length);
		System.arraycopy(b, 0, c, a.length, b.length);
		return c;
	}

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws Exception {
		// # Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
		File f = new File("input.txt");
		if (!f.exists()) {
			try (var in = new URI("https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt").toURL()
					.openStream()) {
				try (var out = new FileOutputStream(f)) {
					out.write(in.readAllBytes());
				}
			}
		}
		List<String> docs = new ArrayList<>(Arrays.asList(new String(Files.readAllBytes(f.toPath())).split("\n")));
		docs.removeIf(String::isBlank);
		docs.replaceAll(String::trim);
		Collections.shuffle(docs, rng);
		System.out.println("num docs: " + docs.size());
		String all = String.join("", docs);
		uchars = all.chars().distinct().sorted().mapToObj(c -> (char) c).toList();
		BOS = uchars.size();
		vocabSize = uchars.size() + 1;
		System.out.println("vocab size: " + vocabSize);
		sd.put("wte", matrix(vocabSize, nEmbd));
		sd.put("wpe", matrix(blockSize, nEmbd));
		sd.put("lm_head", matrix(vocabSize, nEmbd));
		for (int i = 0; i < nLayer; i++) {
			sd.put("layer" + i + ".attn_wq", matrix(nEmbd, nEmbd));
			sd.put("layer" + i + ".attn_wk", matrix(nEmbd, nEmbd));
			sd.put("layer" + i + ".attn_wv", matrix(nEmbd, nEmbd));
			sd.put("layer" + i + ".attn_wo", matrix(nEmbd, nEmbd));
			sd.put("layer" + i + ".mlp_fc1", matrix(4 * nEmbd, nEmbd));
			sd.put("layer" + i + ".mlp_fc2", matrix(nEmbd, 4 * nEmbd));
		}
		System.out.println("num params: " + params.size());

		// Let there be Adam, the blessed optimizer and its buffers
		int nSteps = 1000;
		double lr = 0.01, b1 = 0.85, b2 = 0.99, epsAdam = 1e-8;
		double[] m = new double[params.size()], v = new double[params.size()];

		// Repeat in sequence
		for (int step = 0; step < nSteps; step++) {

			// Take single document, tokenize it, surround it with BOS special token on both sides
			String doc = docs.get(step % docs.size());
			List<Integer> tokens = new ArrayList<>();
			tokens.add(BOS);
			for (char c : doc.toCharArray())
				tokens.add(uchars.indexOf(c));
			tokens.add(BOS);

			// Forward the token sequence through the model, building up the computation graph all the way to the loss
			int n = Math.min(blockSize, tokens.size() - 1);
			List<Value[]>[] keys = new List[nLayer], vals = new List[nLayer];
			for (int i = 0; i < nLayer; i++) {
				keys[i] = new ArrayList<>();
				vals[i] = new ArrayList<>();
			}
			Value loss = Value.of(0);
			for (int pos = 0; pos < n; pos++) {
				Value[] logits = gpt(tokens.get(pos), pos, keys, vals);
				Value[] probs = softmax(logits);
				loss = loss.add(probs[tokens.get(pos + 1)].log().neg());
			}
			loss = loss.div(Value.of(n));
	
			// Backward the loss, calculating the gradients with respect to all model parameters
			loss.backward();
	
			// Adam optimizer update: update the model parameters based on the corresponding gradients
			double lrt = lr * (1.0 - step / (double) nSteps);
			for (int i = 0; i < params.size(); i++) {
				double g = params.get(i).g();
				m[i] = b1 * m[i] + (1 - b1) * g;
				v[i] = b2 * v[i] + (1 - b2) * g * g;
				double mh = m[i] / (1 - Math.pow(b1, step + 1)), vh = v[i] / (1 - Math.pow(b2, step + 1));
				params.get(i).data[0] -= lrt * mh / (Math.sqrt(vh) + epsAdam);
				params.get(i).grad[0] = 0;
			}
			System.out.printf("\rstep %4d/%4d | loss %.4f", step + 1, nSteps, loss.d());
		}

		// Inference: may the model babble back to us
		double temp = 0.5;
		System.out.println("\n--- inference (new, hallucinated names) ---");
		for (int s = 0; s < 20; s++) {
			List<Value[]>[] keys = new List[nLayer], vals = new List[nLayer];
			for (int i = 0; i < nLayer; i++) {
				keys[i] = new ArrayList<>();
				vals[i] = new ArrayList<>();
			}
			int tokenId = BOS;
			StringBuilder sb = new StringBuilder();
			for (int pos = 0; pos < blockSize; pos++) {
				Value[] logits = gpt(tokenId, pos, keys, vals);
				double[] probs = new double[vocabSize];
				Value[] scaled = new Value[vocabSize];
				for (int i = 0; i < vocabSize; i++)
					scaled[i] = Value.of(logits[i].d() / temp);
				Value[] ps = softmax(scaled);
				for (int i = 0; i < vocabSize; i++)
					probs[i] = ps[i].d();
				double r = rng.nextDouble(), cum = 0;
				tokenId = vocabSize - 1;
				for (int i = 0; i < vocabSize; i++) {
					cum += probs[i];
					if (r < cum) {
						tokenId = i;
						break;
					}
				}
				if (tokenId == BOS)
					break;
				sb.append(uchars.get(tokenId));
			}
			System.out.printf("sample %2d: %s%n", s + 1, sb);
		}
	}
}
