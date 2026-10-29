// Auto topics: when a Bot finishes a turn, distil that exchange into short labels
// the sidebar groups its chat by, and remember the seq of the user line each label
// came from, so the Topics tab can point back at the messages. One small call per
// turn on the Bot's own model, asked for the least thinking it offers; a failure
// only warns and the turn simply stays untagged.
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import { splitRef } from './text.js'

export const TOPICS_MAX = 8
export const LABEL_MAX = 8
export const ANCHORS_MAX = 5
const DISTIL_MAX_TOKENS = 256
const BACKFILL_TURNS = 30
const BACKFILL_MAX_TOKENS = 600
const BACKFILL_CHARS = 6000
const DISTIL_TIMEOUT_MS = 60_000
const SETTLE_MS = 2_000
// A turn that ends while another distil for the same Bot still runs is skipped:
// the next one reads the same latest exchange anyway.
const running = new Set()

function parseLabels(reply) {
  const values = []
  const add = value => {
    const label = String(value ?? '').replace(/[#\s]+/g, ' ').trim().replace(/^#+/, '').slice(0, LABEL_MAX)
    if (label !== '' && !values.includes(label)) values.push(label)
  }
  const json = reply.match(/\[[\s\S]*\]/)
  if (json) {
    try { for (const item of JSON.parse(json[0])) add(item) } catch { /* fall through */ }
  }
  if (values.length === 0) for (const line of reply.split(/[\n,，、]/)) add(line)
  return values.slice(0, 3)
}

export function install(rt) {
  const { ctx } = rt

  // The last exchange of a Bot's own chat: the user's line plus the answer lines
  // that follow, walked back through the Session events. Answer lines before the
  // user line (an approval card, say) stay out.
  async function lastExchange(sessionId) {
    const agent = await rt.resolveAgent(sessionId)
    const throughSeq = Number(agent.session.seq) - 1
    const page = await ctx.sessionController.page({
      address: { kind: 'session', sessionId }, throughSeq, maxMessages: 30,
    }, AbortSignal.timeout(20_000))
    const events = page.records.map(record => record.event).reverse()
    const exchange = { prompt: '', promptSeq: 0, promptTurn: undefined, answer: [] }
    let inAnswer = false
    let turn
    for (const event of events) {
      if (event.type === 'turn/start') turn = event.data?.turn
      const entry = rt.historyEntry(event, false)
      if (entry === undefined) continue
      if (entry.answer) {
        if (inAnswer && entry.text !== '') exchange.answer.unshift(entry.text)
        continue
      }
      if (entry.who === 'user' && entry.text !== '') {
        exchange.prompt = entry.text
        exchange.promptSeq = event.seq
        exchange.promptTurn = turn
        break
      }
      inAnswer = true
    }
    return exchange.prompt === '' ? undefined : exchange
  }

  async function modelFor(bot) {
    const route = splitRef(await rt.desiredModel(bot) ?? await rt.defaultModel())
    if (route === undefined) throw new Error('No model is available')
    return route
  }

  async function oneShot(route, system, text, maxTokens = DISTIL_MAX_TOKENS) {
    const info = await ctx.llm.resolveModelInfo(route.provider, route.model).catch(() => undefined)
    const efforts = info?.reasoning?.efforts ?? []
    const effort = efforts.some(entry => entry.id === 'off') ? 'off' : efforts.at(0)?.id
    const assembler = new BlockAssembler()
    for await (const chunk of ctx.llm.stream({
      provider: route.provider, model: route.model,
      ...(effort === undefined ? {} : { reasoningEffort: effort }),
      system, maxTokens,
      messages: [{ role: 'user', content: [{ type: 'text', text }] }],
      signal: AbortSignal.timeout(DISTIL_TIMEOUT_MS),
    })) assembler.push(chunk)
    const finish = assembler.finish
    if (finish.kind === 'error' || finish.kind === 'aborted') throw new Error(finish.failure?.message ?? finish.kind)
    return assembler.blocks().filter(block => block.type === 'text').map(block => block.text).join('\n').trim()
  }

  async function distil(botId) {
    const bot = rt.botOf(botId)
    if (bot === undefined) return
    const exchange = await lastExchange(botId)
    if (exchange === undefined) return
    const language = rt.uiLanguage?.() ?? 'the conversation\'s own language'
    const system = `You label chat exchanges with topic tags. Reply with ONLY a JSON array of 1-3 short topic labels (2-6 characters each) in ${language}, best first. No # prefix, no other text.`
    const text = `User:\n${exchange.prompt.slice(0, 1500)}\n\nAnswer:\n${exchange.answer.join('\n').slice(0, 1000)}`
    const labels = parseLabels(await oneShot(await modelFor(bot), system, text))
    if (labels.length === 0) return
    const topics = Array.isArray(bot.topics) ? [...bot.topics] : []
    const log = { ...bot.topicLog }
    for (const label of labels) {
      if (!topics.includes(label)) topics.push(label)
      const anchor = log[label] ?? { seqs: [], turns: [] }
      anchor.seqs = [exchange.promptSeq, ...anchor.seqs.filter(seq => seq !== exchange.promptSeq)].slice(0, ANCHORS_MAX)
      if (exchange.promptTurn !== undefined) anchor.turns = [exchange.promptTurn, ...anchor.turns.filter(turn => turn !== exchange.promptTurn)].slice(0, ANCHORS_MAX)
      anchor.time = Date.now()
      log[label] = anchor
    }
    // Manual tags the user set keep their place; only the oldest auto ones fall off.
    bot.topics = topics.length > TOPICS_MAX ? topics.slice(topics.length - TOPICS_MAX) : topics
    bot.topicLog = log
    await rt.save()
  }

  ctx.on('agent/status', ({ agent, status }) => {
    if (status !== 'idle' || !rt.isBotSession(agent.id) || running.has(agent.id)) return
    running.add(agent.id)
    // The exchange is read once the turn really settles, not on a status flap.
    setTimeout(() => {
      void distil(agent.id)
        .catch(error => rt.warn('ds-bot: topics for %s failed: %s', agent.id, error?.message ?? error))
        .finally(() => running.delete(agent.id))
    }, SETTLE_MS)
  })

  // Backfill: label a Bot's recent history in one call, for a chat that has turns
  // but no topics yet. The user lines of the last turns go in numbered by seq; the
  // answer maps each label to the seqs it covers, and those become the anchors.
  async function backfill(botId) {
    const bot = rt.botOf(botId)
    if (bot === undefined) return { topics: 0 }
    if (running.has(botId)) return { topics: 0 }
    running.add(botId)
    try {
      // The attempt itself is remembered, so a chat with no user lines (a greeting
      // only, say) is not asked for again on every start.
      if (bot.topicBackfilled !== true) {
        bot.topicBackfilled = true
        await rt.save()
      }
      const agent = await rt.resolveAgent(botId)
      const throughSeq = Number(agent.session.seq) - 1
      const lines = []
      let beforeSeq
      // Pages are walked newest first; the turn/start each user line belongs to
      // still precedes it on the same page or an earlier one.
      let backfillTurn
      for (let pages = 0; pages < 10 && lines.length < BACKFILL_TURNS; pages += 1) {
        const page = await ctx.sessionController.page({
          address: { kind: 'session', sessionId: botId }, throughSeq,
          ...(beforeSeq === undefined ? {} : { beforeSeq }),
          maxMessages: 40,
        }, AbortSignal.timeout(20_000))
        const events = page.records.map(record => record.event)
        for (const event of events) {
          if (event.type === 'turn/start') backfillTurn = event.data?.turn
          const entry = rt.historyEntry(event, false)
          if (entry === undefined || entry.answer) continue
          if (entry.who === 'user' && entry.text !== '') lines.push({ seq: event.seq, turn: backfillTurn, prompt: entry.text })
        }
        if (!page.hasMore || events.length === 0) break
        beforeSeq = events[0].seq
      }
      if (lines.length === 0) return { topics: 0 }
      lines.reverse()
      const language = rt.uiLanguage?.() ?? 'the conversation\'s own language'
      const system = `You label chat histories with topic tags. Reply with ONLY a JSON array of at most ${TOPICS_MAX} objects like {"topic":"label","turns":[n,...]}: topic names the subject matter (a feature, a bug or ticket number, a product area) in 2-6 characters in ${language}, no # prefix; turns lists the line numbers (#n) of the lines that topic covers, most representative first.`
      const text = lines.map(line => `#${line.seq} ${line.prompt.slice(0, 120).replace(/\n+/g, ' ')}`).join('\n').slice(0, BACKFILL_CHARS)
      const reply = await oneShot(await modelFor(bot), system, text, BACKFILL_MAX_TOKENS)
      const known = new Map(lines.map(line => [line.seq, line]))
      let picks
      try { picks = JSON.parse(reply.match(/\[[\s\S]*\]/)?.[0] ?? '[]') } catch { picks = [] }
      const topics = Array.isArray(bot.topics) ? [...bot.topics] : []
      const log = { ...bot.topicLog }
      let added = 0
      for (const pick of picks) {
        const label = String(pick?.topic ?? '').replace(/[#\s]+/g, ' ').trim().slice(0, LABEL_MAX)
        const picked = Array.isArray(pick?.turns) ? [...new Set(pick.turns.map(Number))].filter(seq => known.has(seq)).slice(0, ANCHORS_MAX) : []
        if (label === '' || picked.length === 0) continue
        if (!topics.includes(label)) topics.push(label)
        const anchor = log[label] ?? { seqs: [], turns: [] }
        anchor.seqs = [...picked, ...anchor.seqs.filter(seq => !picked.includes(seq))].slice(0, ANCHORS_MAX)
        // The turn is the unit the official chat's turn rail jumps by; several
        // lines of one turn fold into a single turn anchor.
        const pickedTurns = [...new Set(picked.map(seq => known.get(seq).turn).filter(turn => turn !== undefined))]
        anchor.turns = [...pickedTurns, ...anchor.turns.filter(turn => !pickedTurns.includes(turn))].slice(0, ANCHORS_MAX)
        anchor.time = Date.now()
        log[label] = anchor
        added += 1
      }
      if (added > 0) {
        bot.topics = topics.length > TOPICS_MAX ? topics.slice(topics.length - TOPICS_MAX) : topics
        bot.topicLog = log
        await rt.save()
      }
      return { topics: added }
    } finally {
      running.delete(botId)
    }
  }

  Object.assign(rt, { topicsBackfill: backfill })

  // At start, every Bot with a chat but no topics yet is labelled from its history
  // once, so the sidebar fills in without a panel having to be opened first.
  setTimeout(() => {
    void (async () => {
      try {
        await rt.load()
        for (const bot of Object.values(rt.state.bots ?? {})) {
          if (bot.topicLog !== undefined || bot.topicBackfilled === true) continue
          await backfill(bot.id).catch(error => rt.warn('ds-bot: topics backfill for %s failed: %s', bot.id, error?.message ?? error))
        }
      } catch (error) {
        rt.warn('ds-bot: topics startup pass failed: %s', error?.message ?? error)
      }
    })()
  }, 15_000)
}
