import torch
from .util import text_to_token_ids, token_ids_to_text
from .generate import generate_text_simple, generate
from .gpt2_common import save_training_state
import json

def calc_loss_batch(input_batch, target_batch, model, device="cpu"):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    return loss

def calc_loss_loader(data_loader, model, device="cpu", num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device=device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    device="cpu",
    max_iter=-1,
    skip_steps=-1,
    save_iters=1000,
    stop_sequence=None
):
    loader_length = len(train_loader)
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses = [], []
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            if global_step < skip_steps:
                print(f'Skipping step. {global_step} / {skip_steps}')
                continue

            # Limit for debuggability
            if max_iter > -1 and global_step >= max_iter:
                print(f"Max iterations exceeded at {global_step} steps. Max: {max_iter} steps")
                return train_losses, val_losses

            optimizer.zero_grad() # Reset loss gradients from previous batch iteration

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    eval_iter=eval_iter,
                    device=device
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d} of {loader_length}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
                generate_and_print_sample(
                    model=model,
                    tokenizer=tokenizer,
                    start_context=start_context,
                    device=device,
                    stop_sequence=stop_sequence
                )

            if global_step % save_iters == 0:
                base_model_path = "/home/rngo/code/ttnn-sandbox/notebooks/models"
                model_name = f"checkpoint-model-ep{epoch+1}-{global_step}.pth"
                optimizer_name = f"checkpoint-optimizer-ep{epoch+1}-{global_step}.pth"

                save_training_state(
                    base_model_path=base_model_path,
                    model=model,
                    model_file_name=model_name,
                    optimizer=optimizer,
                    optimizer_file_name=optimizer_name,
                    epoch=epoch,
                    global_step=global_step,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_losses=train_losses,
                    val_losses=val_losses
                )

        # Print a sample text after each epoch
        generate_and_print_sample(
            model=model, 
            tokenizer=tokenizer, 
            start_context=start_context, 
            device=device,
            stop_sequence=stop_sequence
        )

    return train_losses, val_losses

def evaluate_model(model, train_loader, val_loader, eval_iter, device="cpu"):
  model.eval()
  with torch.no_grad():
    train_loss = calc_loss_loader(
        data_loader=train_loader, 
        model=model,
        device=device,
        num_batches=eval_iter
    )
    val_loss = calc_loss_loader(
        data_loader=val_loader,
        model=model,
        device=device,
        num_batches=eval_iter
    )
  model.train()
  return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, start_context, device="cpu", stop_sequence=None):
  model.eval()
  context_size = model.pos_emb.weight.shape[0]
  encoded = text_to_token_ids(start_context, tokenizer).to(device)
  with torch.no_grad():
    token_ids = generate(
        model=model,
        idx=encoded,
        max_new_tokens=100,
        context_size=context_size,
        temperature=1.0,
        top_k=40,
        device=device,
        stop_sequence=stop_sequence
    )
    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=encoded,
    #     max_new_tokens=50,
    #     context_size=context_size
    # )
  decoded_text = token_ids_to_text(token_ids, tokenizer)
  print(decoded_text.replace("\n", " "))  # Compact print format
  model.train()
  