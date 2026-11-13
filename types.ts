export type IntelligenceMode = 'quick' | 'normal' | 'deep'

export type IntelligenceRequest = {
  conversationId?: string
  message: string
  mode?: IntelligenceMode
  projectId?: string
}

export type IntelligenceResponse = {
  text: string
  source: 'tensorflow' | 'mentor' | 'orchestrator'
  modelReady: boolean
}

export interface OriIntelligenceProvider {
  respond(request: IntelligenceRequest): Promise<IntelligenceResponse>
}
