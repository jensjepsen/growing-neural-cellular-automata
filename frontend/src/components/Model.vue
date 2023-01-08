<script setup lang="ts">
import { ref, onMounted, Ref, reactive, watch, toRef, defineProps } from 'vue'

import { run, drawNext, damage } from '../helpers/main'

// Define props
const props = defineProps({ modelFile: String, title: String })

// Reactive state
const state = reactive({
  iterations: 0
})

// Refs
let canvas: Ref<HTMLCanvasElement | null>  = ref(null);
let context: CanvasRenderingContext2D | null | undefined = undefined;
let timeoutHandle: number = 0;

watch([toRef(props, 'modelFile')], () => {
  ort.InferenceSession.create(`./models/${props.modelFile}`).then((session) => {
    if(canvas.value && context && props.modelFile) {
      loop(null, session, canvas.value, context, props.modelFile)
      reset = true
    }
  })
}, {immediate: true})

let click: MouseEvent | null = null
let reset = false

function loop(cellState: Float32Array | null, session: ort.InferenceSession, canvas: HTMLCanvasElement, context: CanvasRenderingContext2D, modelFile: string) {
    if (modelFile !== props.modelFile) {
      return
    }

    if (reset) {
      reset = false
      cellState = null
      state.iterations = 0
    }
    cellState = cellState || new Float32Array(1 * 40 * 40 * 16);
    const iters = 1
    run(iters, cellState, session).then((result) => {
        //drawNext(result.rgb, 0)
        const nextState = result.last_state
        if (!!click) {
            damage(nextState, {y: Math.round(40 * click.offsetY / (<any>click.target).clientHeight), x: Math.round(40 * click.offsetX / (<any>click.target).clientWidth)}, 10)
            click = null
        }
        drawNext(
            result.rgb,
            0,
            () => loop(nextState.data, session, canvas, context, modelFile),
            canvas,
            context
        )
        state.iterations += iters
    })
}

// Event handlers
function canvasClick (e: MouseEvent) {
  click = e
}

function doReset () {
  reset = true
}



// Lifecycle hooks
onMounted(() => {
  context = canvas.value?.getContext('2d')
})
</script>

<template>
  <div class="model-container">
      <div class="is-flex is-justify-content-space-around is-size-6"><b>{{ title }}</b></div>
      <div class="is-flex is-justify-content-space-around">
        <canvas class="canvas" width="40" height="40" ref="canvas" @click="canvasClick">
        </canvas>
      </div>
      <div class="is-flex is-flex-direction-row is-justify-content-space-between">
        <div class="is-size-7 iters is">{{state.iterations}}</div>
        <button class="button is-small" @click="doReset()">
          reset
        </button>
      </div>
  </div>
</template>

<style scoped>
  .canvas {
      width: 200px;
      height: 200px;
      cursor: pointer;
  }
</style>
