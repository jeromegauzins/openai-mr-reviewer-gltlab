import {info, setFailed, warning} from './gitlab-core'
import {OpenAIOptions, Options} from './options'

import {createOpenAI, openai, OpenAIProvider} from '@ai-sdk/openai'
import {generateText, GenerateTextResult} from 'ai'

export class Bot {
  private readonly aiProvider: OpenAIProvider | null = null // not free
  private readonly model: ReturnType<typeof openai> | null = null

  private readonly options: Options

  constructor(options: Options, openaiOptions: OpenAIOptions) {
    this.options = options

    if (process.env.OPENAI_API_KEY) {
      const currentDate = new Date().toISOString().split('T')[0]
      const systemMessage = `${options.systemMessage}
Knowledge cutoff: ${openaiOptions.tokenLimits.knowledgeCutOff}
Current date: ${currentDate}`

      this.aiProvider = createOpenAI({
        apiKey: process.env.OPENAI_API_KEY,
        organization: process.env.OPENAI_API_ORG ?? undefined,
        baseURL: options.apiBaseUrl,
        compatibility: 'strict' // strict mode, enable when using the OpenAI API
      })

      this.model = openai('gpt-4o', {user: process.env.CI_COMMIT_AUTHOR})
    } else {
      const err =
        "Unable to initialize the OpenAI API, 'OPENAI_API_KEY' environment variable are not available"
      throw new Error(err)
    }
  }

  chat = async (message: string): Promise<string> => {
    let res = ''
    try {
      res = await this.chat_(message)
      return res
    } catch (e: unknown) {
      if (e instanceof Error) {
        warning(`Failed to chat: ${e}, backtrace: ${e.stack}`)
      }
      return res
    }
  }

  private readonly chat_ = async (message: string): Promise<string> => {
    // record timing
    const start = Date.now()
    if (!message) {
      return ''
    }

    let response: GenerateTextResult<any> | undefined
    if (!this.aiProvider || !this.model) {
      setFailed('The OpenAI API is not initialized')
    } else {
      try {
        response = await generateText({
          maxRetries: this.options.openaiRetries,

          // TODO: timeout
          prompt: message,
          system: this.options.systemMessage,
          model: this.model
        })
      } catch (e: unknown) {
        if (e instanceof Error) {
          info(
            `response: ${response}, failed to send message to openai: ${e}, backtrace: ${e.stack}`
          )
        }
      }
      const end = Date.now()
      info(`response: ${JSON.stringify(response)}`)
      info(
        `openai sendMessage (including retries) response time: ${
          end - start
        } ms`
      )
    }

    let responseText = ''
    if (response != null) {
      responseText = response.text
    } else {
      warning('openai response is null')
    }
    // remove the prefix "with " in the response
    if (responseText.startsWith('with ')) {
      responseText = responseText.substring(5)
    }
    if (this.options.debug) {
      info(`openai responses: ${responseText}`)
    }
    return responseText
  }
}
