## Omi Agent Plugin

Omi Agent is a community-built plugin that lets your Omi device listen for trigger phrases like **“Omi”**, understands your question, and respond naturally.

This plugin uses a simple webhook to process voice input and send replies through Omi notifications, giving you a hands-free conversational experience.

### How it works

1. Omi streams real-time speech to your webhook endpoint.  
2. The plugin detects trigger phrases such as **“Omi”**.  
3. The spoken question is transcribed and sent for processing.  
4. A short, relevant reply is generated and sent back to Omi.  

You can customize this behavior by editing the webhook logic in your server.

### Quick Deploy

You can deploy your own version easily with:

```bash
git clone https://github.com/rojansapkota/omi-plugins
cd omi-agent
pip install -r requirements.txt
uvicorn main:app --reload
```

Then copy your server URL (e.g. `https://your-domain.com/webhook`) into the **Omi App** webhook field.

---

## Example Response

> **You:** "Omi, What's the Capital of USA"  
> **Omi:** "The capital of the United States of America is Washington, D.C. (short for District of Columbia)."

---

## License

This plugin is released under the **MIT License**, following the spirit of openness in the Omi ecosystem.


## Contributions

- Join the [Discord](http://discord.omi.me).
- Build your own [Plugins/Integrations](https://docs.omi.me/doc/developer/apps/Introduction).