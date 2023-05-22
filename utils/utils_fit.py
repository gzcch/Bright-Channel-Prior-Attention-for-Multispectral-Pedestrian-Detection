import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def E_loss(t_pt, t_p, t_pmin, lam=0.1, sigma=0.5, alpha=0.5, kernel_size=3):
    l_bcp1 = torch.sum((t_pt - t_p) ** 2)
    unfold_tp = torch.nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)(t_p)  # (b,k*k,p(patches num))
    l_bcp2 = 0
    for i in range(kernel_size * kernel_size):
        for j in range(i):
            tem = (unfold_tp[:, i, :] - unfold_tp[:, j, :]).pow(2)
            l_bcp2 += torch.sum(torch.exp(-sigma * tem) * (tem))

    lam = 0.1
    L_BCP = torch.sum((l_bcp1 + lam * l_bcp2) / (t_p.shape[2] * t_p.shape[3]))
    M = torch.clamp(t_p - t_pmin, min=0.0)
    L_S = torch.sum((M * (t_p - t_pmin)).pow(2)) / (t_p.shape[2] * t_p.shape[3])

    return L_BCP + alpha * L_S


def fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    #model.load_state_dict(torch.load('logs/best_epoch_weights.pth'))
    model_train.train()
    #eval_callback.on_epoch_end(epoch + 1, model_train)

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, images_T, targets = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                images_T = images_T.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        # ----------------------#
        #   清零梯度
        # ----------------------#
        with torch.no_grad():
            images_T = images_T.unsqueeze(dim=1)
            images_F = torch.cat((images, images_T), dim=1)
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            torch.cuda.empty_cache()
            torch.cuda.max_memory_allocated = 2 * 1024 * 1024 * 1024
            outputs = model_train(images_F)

            loss_value_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            for l in range(len(outputs)-3):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            Eloss = 0.5 *  E_loss(outputs[3], outputs[4], outputs[5], images)
            loss_value = loss_value_all + Eloss

            # ----------------------#
            #   反向传播
            # ----------------------#
            torch.cuda.empty_cache()
            loss_value.backward()
            optimizer.step()

        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(images_F)

                loss_value_all = 0
                # ----------------------#
                #   计算损失
                # ----------------------#
                for l in range(len(outputs)-3):
                    with torch.cuda.amp.autocast(enabled=False):
                        predication = outputs[l].float()
                    loss_item = yolo_loss(l, predication, targets)
                    loss_value_all += loss_item
                Eloss = 0.5 * E_loss(outputs[3], outputs[4], outputs[5], images)



                loss_value = loss_value_all + Eloss

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

        if iteration % 50 == 0:
            print('\n yolo Loss: %.3f || E Loss: %.3f ' % (loss_value_all , Eloss ))


    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, images_T, targets = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                images_T = images_T.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            images_T = images_T.unsqueeze(dim=1)
            images_F = torch.cat((images, images_T), dim=1)

            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(images_F)

            loss_value_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            for l in range(len(outputs) - 3):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)



    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
