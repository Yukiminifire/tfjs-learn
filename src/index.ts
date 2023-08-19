import {tensor,scalar, variable, zeros} from "@tensorflow/tfjs"
import "@tensorflow/tfjs-node"

const t = zeros([5,5])
const v = variable(t)
console.log(v)
v.print()
t.print()